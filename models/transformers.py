from math import pi, log
from functools import wraps
from multiprocessing import context
from textwrap import indent
import models.util_funcs as util_funcs
import math, copy
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import pdb
from einops.layers.torch import Rearrange
from options import get_parser_main_model
opts = get_parser_main_model().parse_args()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    '''
    x: ([64, 64, 2, 1]) is between [-1,1]
    max_feq is 10
    num_bands is 6
    '''
    
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype) # tensor([1.0000, 1.8000, 2.6000, 3.4000, 4.2000, 5.0000]
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)] # r([[[[1.0000, 1.8000, 2.6000, 3.4000, 4.2000, 5.0000]]]],

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    
    x = torch.cat((x, orig_x), dim = -1)
    return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.,cls_conv_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False) # 27 to 5012*2 = 1024

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)
        #self.cls_dim_adjust = nn.Linear(context_dim,cls_conv_dim)

    def forward(self, x, context = None, mask = None, ref_cls_onehot=None):

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask): 
            mask = repeat(mask, 'b j k -> (b h) k j', h = h)
            sim.masked_fill(mask == 0, -1e9)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out), attn


class SVGEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_embed = nn.Embedding(4, 512)
        self.arg_embed = nn.Embedding(128, 128,padding_idx=0)
        self.embed_fcn = nn.Linear(128 * 8, 512)
        self.pos_encoding = PositionalEncoding(d_model=opts.hidden_size, max_len=opts.max_seq_len + 1)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")


    def forward(self, commands, args, groups=None):
        
        S, GN,_ = commands.shape 
        src = self.command_embed(commands.long()).squeeze() + \
            self.embed_fcn(self.arg_embed((args).long()).view(S, GN, -1)) # shift due to -1 PAD_VAL

        src = self.pos_encoding(src)

        return src
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(F.relu(self.dropout(self.w_1(x))))

class Transformer_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.SVG_embedding = SVGEmbedding()
        self.command_fcn = nn.Linear(512, 4)
        self.args_fcn = nn.Linear(512, 8 * 128)
        c = copy.deepcopy
        attn = MultiHeadedAttention(h=8, d_model=512, dropout=0.0)
        ff = PositionwiseFeedForward(d_model=512, d_ff=1024, dropout=0.0)
        self.decoder_layers = clones(DecoderLayer(512, c(attn), c(attn),c(ff), dropout=0.0), 6)
        self.decoder_norm = nn.LayerNorm(512)
        self.decoder_layers_parallel = clones(DecoderLayer(512, c(attn), c(attn), c(ff), dropout=0.0), 1)
        self.decoder_norm_parallel = nn.LayerNorm(512)
        self.cls_embedding = nn.Embedding(52,512)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))

    def forward(self, x, memory, trg_char, src_mask=None, tgt_mask=None):

        memory = memory.unsqueeze(1)
        commands = x[:, :, :1]
        args = x[:, :, 1:]
        x = self.SVG_embedding(commands, args).transpose(0,1)
        trg_char = trg_char.long()
        trg_char = self.cls_embedding(trg_char)
        x[:, 0:1, :] = trg_char 
        tgt_mask = tgt_mask.squeeze()
        for layer in self.decoder_layers:
            x,attn = layer(x, memory, src_mask, tgt_mask) 
        out = self.decoder_norm(x)
        N, S, _ = out.shape
        cmd_logits = self.command_fcn(out)
        args_logits = self.args_fcn(out) # shape: bs, max_len, 8, 256
        args_logits = args_logits.reshape(N, S, 8, 128)
        return cmd_logits,args_logits,attn
    
    def parallel_decoder(self, cmd_logits, args_logits, memory, trg_char):

        memory = memory.unsqueeze(1)
        cmd_args_mask =  torch.Tensor([[0, 0, 0., 0., 0., 0., 0., 0.],
                                       [1, 1, 0., 0., 0., 0., 1., 1.],
                                       [1, 1, 0., 0., 0., 0., 1., 1.],
                                       [1, 1, 1., 1., 1., 1., 1., 1.]]).to(cmd_logits.device)  
        if opts.mode == 'train':
            cmd2 = torch.argmax(cmd_logits, -1).unsqueeze(-1).transpose(0, 1) 
            arg2 = torch.argmax(args_logits, -1).transpose(0, 1)

            cmd2paddingmask = _get_key_padding_mask(cmd2).transpose(0,1).unsqueeze(-1).to(cmd2.device)  
            cmd2 = cmd2 * cmd2paddingmask
            args_mask = torch.matmul(F.one_hot(cmd2.long(),4).float(), cmd_args_mask).transpose(-1,-2).squeeze(-1)
            arg2 = arg2 * args_mask     

            x = self.SVG_embedding(cmd2, arg2).transpose(0, 1)
        else:
            cmd2 = cmd_logits
            arg2 = args_logits

            cmd2paddingmask = _get_key_padding_mask(cmd2).transpose(0, 1).unsqueeze(-1).to(cmd2.device)
            cmd2 = cmd2 * cmd2paddingmask
            args_mask = torch.matmul(F.one_hot(cmd2.long(),4).float(), cmd_args_mask).transpose(-1, -2).squeeze(-1) 
            arg2 = arg2 * args_mask

            x = self.SVG_embedding(cmd2, arg2).transpose(0,1)

        S = x.size(1)
        B = x.size(0)
        tgt_mask = torch.ones(S,S).to(x.device).unsqueeze(0).repeat(B, 1, 1)
        cmd2paddingmask = cmd2paddingmask.transpose(0, 1).transpose(-1, -2)
        tgt_mask  = tgt_mask * cmd2paddingmask

        trg_char = trg_char.long()
        trg_char = self.cls_embedding(trg_char)
        
        x = torch.cat([trg_char, x],1)
        x[:, 0:1, :] = trg_char
        x = x[:,:opts.max_seq_len,:]
        tgt_mask = tgt_mask #*tri
        for layer in self.decoder_layers_parallel:
            x, attn = layer(x, memory, src_mask=None, tgt_mask=tgt_mask)
        out = self.decoder_norm_parallel(x)

        N, S, _ = out.shape
        cmd_logits = self.command_fcn(out)
        args_logits = self.args_fcn(out)
        args_logits = args_logits.reshape(N, S, 8, 128)

        return cmd_logits, args_logits


def _get_key_padding_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    lens =[]
    with torch.no_grad():
        key_padding_mask = (commands == 0).cumsum(dim=seq_dim) > 0
        commands=commands.transpose(0,1).squeeze(-1) #bs, opts.max_seq_len
        for i in range(commands.size(0)):
            try:
                seqi = commands[i]#blue opts.max_seq_len
                index = torch.where(seqi==0)[0][0]
               
            except:
                index=opts.max_seq_len
                
            lens.append(index)
        lens = torch.tensor(lens)+1#blue b
        seqlen_mask = util_funcs.sequence_mask(lens, opts.max_seq_len)#blue b,opts.max_seq_len
        return seqlen_mask

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 1,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 2,
        final_classifier_head = True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0 # 26
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))


        #self_per_cross_attn=1
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn): #BUG 之前是2  self_per_cross_attn
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))


        get_cross_attn2 = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff2 = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn2 = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff2 = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn2, get_cross_ff2, get_latent_attn2, get_latent_ff2 = map(cache_fn, (get_cross_attn2, get_cross_ff2, get_latent_attn2, get_latent_ff2))

        self.layers_cnnsvg = nn.ModuleList([])
        for i in range(1):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns2 = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn): 
                self_attns2.append(nn.ModuleList([
                    get_latent_attn2(**cache_args, key = block_ind),
                    get_latent_ff2(**cache_args, key = block_ind)
                ]))

            self.layers_cnnsvg.append(nn.ModuleList([
                get_cross_attn2(**cache_args),
                get_cross_ff2(**cache_args),
                self_attns2
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()
        self.pre_lstm_fc = nn.Linear(10,opts.hidden_size)
        self.posr = PositionalEncoding(d_model=opts.hidden_size,max_len=opts.max_seq_len)
        
        patch_height = 2
        patch_width = 2
        patch_dim =  1 * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, 16),
        )

        self.SVG_embedding = SVGEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))

    def forward(self, data, seq, ref_cls_onehot=None, mask=None, return_embeddings=True):

        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis' # img is 2
        x = seq
        commands=x[:, :, :1]
        args=x[:, :, 1:]
        x = self.SVG_embedding(commands, args).transpose(0,1)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x.size(0))
        x = torch.cat([cls_tokens,x],dim = 1)
        cls_one_pad = torch.ones((1,1,1)).to(x.device).repeat(x.size(0),1,1)
        mask = torch.cat([cls_one_pad,mask],dim=-1)
        self_atten = []
        for cross_attn, cross_ff, self_attns in self.layers:
            for self_attn, self_ff in self_attns: 
                x_,atten = self_attn(x,mask=mask)
                x = x_ + x 
                self_atten.append(atten)
                x = self_ff(x) + x
        x = x + torch.randn_like(x) # add a perturbation
        return x, self_atten
    
    def att_residual(self, x, mask=None):

        for cross_attn, cross_ff, self_attns in self.layers_cnnsvg:
            for self_attn, self_ff in self_attns: 
                x_, atten = self_attn(x)  
                x = x_ + x 
                x = self_ff(x) + x
        return x

        

    def loss(self, cmd_logits, args_logits, trg_seq, trg_seqlen, trg_pts_aux):
        '''
        Inputs:
        cmd_logits: [b, 51, 4]
        args_logits: [b, 51, 6]
        '''
        cmd_args_mask =  torch.Tensor([[0, 0, 0., 0., 0., 0., 0., 0.],
                                       [1, 1, 0., 0., 0., 0., 1., 1.],
                                       [1, 1, 0., 0., 0., 0., 1., 1.],
                                       [1, 1, 1., 1., 1., 1., 1., 1.]]).to(cmd_logits.device)  
        
        tgt_commands = trg_seq[:,:,:1].transpose(0,1)
        tgt_args = trg_seq[:,:,1:].transpose(0,1)
        
        seqlen_mask = util_funcs.sequence_mask(trg_seqlen, opts.max_seq_len).unsqueeze(-1)
        seqlen_mask2 = seqlen_mask.repeat(1,1,4)# NOTE b,501,4
        seqlen_mask4 = seqlen_mask.repeat(1,1,8)
        seqlen_mask3 = seqlen_mask.unsqueeze(-1).repeat(1,1,8,128)
        
        
        tgt_commands_onehot = F.one_hot(tgt_commands, 4)
        tgt_args_onehot = F.one_hot(tgt_args, 128)
       
        args_mask = torch.matmul(tgt_commands_onehot.float(),cmd_args_mask).squeeze()


        loss_cmd = torch.sum(- tgt_commands_onehot.squeeze() * F.log_softmax(cmd_logits, -1), -1)
        loss_cmd = torch.mul(loss_cmd, seqlen_mask.squeeze())
        loss_cmd = torch.mean(torch.sum(loss_cmd/trg_seqlen.unsqueeze(-1),-1))
        
        loss_args = (torch.sum(-tgt_args_onehot*F.log_softmax(args_logits,-1),-1)*seqlen_mask4*args_mask)
     
        loss_args = torch.mean(loss_args,dim=-1,keepdim=False)
        loss_args = torch.mean(torch.sum(loss_args/trg_seqlen.unsqueeze(-1),-1))

        SE_mask =  torch.Tensor([[1, 1],
                                 [0, 0],
                                 [1, 1],
                                 [1, 1]]).to(cmd_logits.device)  
        
        SE_args_mask = torch.matmul(tgt_commands_onehot.float(),SE_mask).squeeze().unsqueeze(-1)
        
        
        args_prob = F.softmax(args_logits, -1)
        args_end = args_prob[:,:,6:]
        args_end_shifted = torch.cat((torch.zeros(args_end.size(0),1,args_end.size(2),args_end.size(3)).to(args_end.device),args_end),1)
        args_end_shifted = args_end_shifted[:,:opts.max_seq_len,:,:]
        args_end_shifted = args_end_shifted*SE_args_mask + args_end*(1-SE_args_mask)
        
        args_start = args_prob[:,:,:2]

        seqlen_mask5 = util_funcs.sequence_mask(trg_seqlen-1, opts.max_seq_len).unsqueeze(-1)
        seqlen_mask5 = seqlen_mask5.repeat(1,1,2)
       
        smooth_constrained = torch.sum(torch.pow((args_end_shifted - args_start), 2), -1) * seqlen_mask5
        smooth_constrained = torch.mean(smooth_constrained, dim=-1, keepdim=False)
        smooth_constrained = torch.mean(torch.sum(smooth_constrained / (trg_seqlen - 1).unsqueeze(-1), -1))

        args_prob2 = F.softmax(args_logits / 0.1, -1)

        c = torch.argmax(args_prob2,-1).unsqueeze(-1).float() - args_prob2.detach()
        p_argmax = args_prob2 + c
        p_argmax = torch.mean(p_argmax,-1)       
        control_pts = denumericalize(p_argmax)
        
        p0 = control_pts[:,:,:2]
        p1 = control_pts[:,:,2:4]
        p2 = control_pts[:,:,4:6]
        p3 = control_pts[:,:,6:8]
        
        line_mask = (tgt_commands==2).float() + (tgt_commands==1).float() 
        curve_mask = (tgt_commands==3).float() 
        
        t=0.25
        aux_pts_line = p0 + t*(p3-p0)
        for t in [0.5,0.75]:
            coord_t = p0 + t*(p3-p0)
            aux_pts_line = torch.cat((aux_pts_line,coord_t),-1)
        aux_pts_line = aux_pts_line*line_mask
        
        t=0.25
        aux_pts_curve = (1-t)*(1-t)*(1-t)*p0 + 3*t*(1-t)*(1-t)*p1 + 3*t*t*(1-t)*p2 + t*t*t*p3
        for t in [0.5, 0.75]:
            coord_t = (1-t)*(1-t)*(1-t)*p0 + 3*t*(1-t)*(1-t)*p1 + 3*t*t*(1-t)*p2 + t*t*t*p3
            aux_pts_curve = torch.cat((aux_pts_curve,coord_t),-1)
        aux_pts_curve = aux_pts_curve * curve_mask
        
        
        aux_pts_predict = aux_pts_curve + aux_pts_line
        seqlen_mask_aux = util_funcs.sequence_mask(trg_seqlen - 1, opts.max_seq_len).unsqueeze(-1)
        aux_pts_loss = torch.pow((aux_pts_predict - trg_pts_aux), 2) * seqlen_mask_aux
        
        loss_aux = torch.mean(aux_pts_loss, dim=-1, keepdim=False)
        loss_aux = torch.mean(torch.sum(loss_aux / trg_seqlen.unsqueeze(-1), -1))

        
        loss = opts.loss_w_cmd * loss_cmd + opts.loss_w_args * loss_args + opts.loss_w_aux * loss_aux + opts.loss_w_smt * smooth_constrained 

        svg_losses = {}
        svg_losses['loss_total'] = loss
        svg_losses["loss_cmd"] = loss_cmd
        svg_losses["loss_args"] = loss_args
        svg_losses["loss_smt"] = smooth_constrained
        svg_losses["loss_aux"] = loss_aux

        return svg_losses
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        attn = self.self_attn.attn
        return self.sublayer[2](x, self.feed_forward),attn

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def numericalize(cmd, n=128):
    """NOTE: shall only be called after normalization"""
    # assert np.max(cmd.origin) <= 1.0 and np.min(cmd.origin) >= -1.0 
    cmd = (cmd / 30 * n).round().clip(min=0, max=n-1).int()
    return cmd

def denumericalize(cmd, n=128):
    cmd = cmd / n * 30 
    return cmd

def attention(query, key, value, mask=None, trg_tri_mask=None,dropout=None, posr=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if posr is not None:
        posr = posr.unsqueeze(1)
        scores = scores + posr

    if mask is not None:
        try:
            scores = scores.masked_fill(mask == 0, -1e9) # note mask: b,1,501,501  scores: b, head, 501,501
        except:
            pdb.set_trace()

    if trg_tri_mask is not None:
        scores = scores.masked_fill(trg_tri_mask == 0, -1e9) 
    
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h #32
        self.h = h #8
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None,trg_tri_mask=None, posr=None):
        "Implements Figure 2"

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0) #16

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,trg_tri_mask=trg_tri_mask,
                                 dropout=self.dropout, posr=posr)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x_norm=self.norm(x)
        return x + self.dropout(sublayer(x_norm))#+ self.augs(x_norm)


if __name__ == '__main__':
    model = Transformer(
        input_channels = 1,          # number of channels for each token of the input
        input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
        max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                    #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = 512,            # latent dimension
        cross_heads = 1,             # number of heads for cross attention. paper said 1
        latent_heads = 8,            # number of heads for latent self attention, 8
        cross_dim_head = 64,         # number of dimensions per cross attention head
        latent_dim_head = 64,        # number of dimensions per latent self attention head
        num_classes = 1000,          # output number of classes
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn = 2      # number of self attention blocks per cross attention
    )
    
    img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized

    model(img) # (1, 1000)