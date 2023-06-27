import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from options import get_parser_main_model
opts = get_parser_main_model().parse_args()

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class ModalityFusion(nn.Module):
    def __init__(self, img_size=64, ref_nshot=4, bottleneck_bits=512, ngf=32, seq_latent_dim=512, mode='train'):
        super().__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.ref_nshot = ref_nshot
        self.mode = mode
        self.fc_merge = nn.Linear(seq_latent_dim * opts.ref_nshot, 512)
        n_downsampling = int(math.log(img_size, 2))
        mult_max = 2 ** (n_downsampling)
        self.fc_fusion = nn.Linear(ngf * mult_max + seq_latent_dim, opts.bottleneck_bits * 2, bias=True) # the max multiplier for img feat channels is 

    def forward(self, seq_feat, img_feat, ref_pad_mask=None):


        cls_one_pad = torch.ones((1,1,1)).to(seq_feat.device).repeat(seq_feat.size(0),1,1)
        ref_pad_mask = torch.cat([cls_one_pad,ref_pad_mask],dim=-1)
        
        seq_feat = seq_feat * (ref_pad_mask.transpose(1, 2))
        seq_feat_ = seq_feat.view(seq_feat.size(0) // self.ref_nshot, self.ref_nshot,seq_feat.size(-2) , seq_feat.size(-1))
        seq_feat_ = seq_feat_.transpose(1, 2)
        seq_feat_ = seq_feat_.contiguous().view(seq_feat_.size(0), seq_feat_.size(1), seq_feat_.size(2) * seq_feat_.size(3))
        seq_feat_ = self.fc_merge(seq_feat_)
        seq_feat_cls = seq_feat_[:, 0]

        feat_cat = torch.cat((img_feat, seq_feat_cls),-1)
        dist_param = self.fc_fusion(feat_cat)

        output = {}
        mu = dist_param[..., :self.bottleneck_bits]
        log_sigma = dist_param[..., self.bottleneck_bits:]

        if self.mode == 'train':
            # calculate the kl loss and reparamerize latent code
            epsilon = torch.randn(*mu.size(), device=mu.device)
            z = mu + torch.exp(log_sigma / 2) * epsilon
            kl = 0.5 * torch.mean(torch.exp(log_sigma) + torch.square(mu) - 1. - log_sigma)
            output['latent'] = z
            output['kl_loss'] = kl
            seq_feat_[:, 0] = z
            latent_feat_seq = seq_feat_

        else:
            output['latent'] = mu
            output['kl_loss'] = 0.0
            seq_feat_[:, 0] = mu
            latent_feat_seq = seq_feat_

        
        return output, latent_feat_seq


