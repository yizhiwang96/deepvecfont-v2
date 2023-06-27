import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImageEncoder(nn.Module):

    def __init__(self, img_size, input_nc, ngf=16, norm_layer=nn.LayerNorm):

        super(ImageEncoder, self).__init__()
        n_downsampling = int(math.log(img_size, 2))
        ks_list = [5] * (n_downsampling - n_downsampling // 3) + [3] * (n_downsampling // 3)
        stride_list = [2] * n_downsampling

        chn_mult = []
        for i in range(n_downsampling):
            chn_mult.append(2 ** (i + 1))
        
        encoder = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=7 // 2, bias=True, padding_mode='replicate'),
                   norm_layer([ngf, 2 ** n_downsampling, 2 ** n_downsampling]),
                   nn.ReLU(True)]
        for i in range(n_downsampling):  # add downsampling layers
            if i == 0:
                chn_prev = ngf
            else:
                chn_prev = ngf * chn_mult[i - 1]
            chn_next = ngf * chn_mult[i]

            encoder += [nn.Conv2d(chn_prev, chn_next, kernel_size=ks_list[i], stride=stride_list[i], padding=ks_list[i] // 2, padding_mode='replicate'),
                        norm_layer([chn_next, 2 ** (n_downsampling - 1 - i), 2 ** (n_downsampling - 1 - i)]),
                        nn.ReLU(True)]

        self.encode = nn.Sequential(*encoder)
        self.flatten = nn.Flatten()
 
    def forward(self, input):
        """Standard forward"""
        ret = self.encode(input)    
        img_feat = self.flatten(ret)
        output = {}
        output['img_feat'] = img_feat
        return output
