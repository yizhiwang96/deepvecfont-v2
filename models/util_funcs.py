import torch
import torch.nn.functional as F
import cairosvg
from data_utils.common_utils import trans2_white_bg
from PIL import Image
import numpy as np

def select_imgs(images_of_onefont, selected_cls, opts):
    # given selected char classes, return selected imgs
    # images_of_onefont: [bs, 52, opts.img_size, opts.img_size]
    # selected_cls: [bs, nshot]
    nums = selected_cls.size(1)
    selected_cls_ = selected_cls.unsqueeze(2)
    selected_cls_ = selected_cls_.unsqueeze(3)
    selected_cls_ = selected_cls_.expand(images_of_onefont.size(0), nums, opts.img_size, opts.img_size)         
    selected_img = torch.gather(images_of_onefont, 1, selected_cls_)
    return selected_img

def select_seqs(seqs_of_onefont, selected_cls, opts, seq_dim):

    nums = selected_cls.size(1)
    selected_cls_ = selected_cls.unsqueeze(2)
    selected_cls_ = selected_cls_.unsqueeze(3)
    selected_cls_ = selected_cls_.expand(seqs_of_onefont.size(0), nums, opts.max_seq_len, seq_dim) 
    selected_seqs = torch.gather(seqs_of_onefont, 1, selected_cls_)
    return selected_seqs

def select_seqlens(seqlens_of_onefont, selected_cls, opts):

    nums = selected_cls.size(1)
    selected_cls_ = selected_cls.unsqueeze(2)
    selected_cls_ = selected_cls_.expand(seqlens_of_onefont.size(0), nums, 1)     # 64, nums, 1    
    selected_seqlens = torch.gather(seqlens_of_onefont, 1, selected_cls_)
    return selected_seqlens

def trgcls_to_onehot(trg_cls, opts):
    trg_char = F.one_hot(trg_cls, num_classes=opts.char_num).squeeze(dim=1)
    return trg_char


def shift_right(x, pad_value=None):
    if pad_value is None:
        shifted = F.pad(x, (0, 0, 0, 0, 1, 0))[:-1, :, :]
    else:
        shifted = torch.cat([pad_value, x], axis=0)[:-1, :, :]
    return shifted


def length_form_embedding(emb):
    """Compute the length of each sequence in the batch
    Args:
        emb: [seq_len, batch, depth]
    Returns:
        a 0/1 tensor: [batch]
    """
    absed = torch.abs(emb)
    sum_last = torch.sum(absed, dim=2, keepdim=True)
    mask = sum_last != 0
    sum_except_batch = torch.sum(mask, dim=(0, 2), dtype=torch.long)
    return sum_except_batch


def lognormal(y, mean, logstd, logsqrttwopi):
    y_mean = y - mean # NOTE y:[b*51*6, 1]   mean:  [b*51*6, 50]
    logstd_exp = logstd.exp() # NOTE  [b*51*6, 50]
    y_mean_divide_exp = y_mean / logstd_exp
    return -0.5 * (y_mean_divide_exp) ** 2 - logstd - logsqrttwopi

def sequence_mask(lengths, max_len=None):
    batch_size=lengths.numel()
    max_len=max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size,max_len)
    .lt(lengths.unsqueeze(1)))

def svg2img(path_svg, path_img, img_size):
    cairosvg.svg2png(url=path_svg, write_to=path_img, output_width=img_size, output_height=img_size)
    img_arr = trans2_white_bg(path_img)
    return img_arr

def cal_img_l1_dist(path_img1, path_img2):
    img1 = np.array(Image.open(path_img1))
    img2 = np.array(Image.open(path_img2))
    dist = np.mean(np.abs(img1 - img2[:, :, 0]))
    return dist

def cal_iou(path_img1, path_img2):

    img1 = np.array(Image.open(path_img1))
    img2 = np.array(Image.open(path_img2))[:, :, 0]
    mask_img1 = img1 < (255 * 3 / 4)
    mask_img2 = img2 < (255 * 3 / 4)
    iou = np.sum(mask_img1 * mask_img2) / (np.sum(mask_img1 + mask_img2))
    l1_dist = np.mean(np.abs(mask_img1.astype(float) - mask_img2.astype(float)))
    return iou, l1_dist