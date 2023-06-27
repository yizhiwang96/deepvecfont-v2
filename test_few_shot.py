import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from dataloader import get_loader
from models.model_main import ModelMain
from models.transformers import denumericalize
from options import get_parser_main_model
from data_utils.svg_utils import render
from models.util_funcs import svg2img, cal_iou

def test_main_model(opts):

    dir_res = os.path.join("./experiments/", opts.name_exp, "results")

    test_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num, opts.max_seq_len, opts.dim_seq, opts.batch_size, 'test')

    model_main = ModelMain(opts)
    path_ckpt = os.path.join('experiments', opts.name_exp, 'checkpoints', opts.name_ckpt)
    model_main.load_state_dict(torch.load(path_ckpt)['model'])
    model_main.cuda()
    model_main.eval()

    with torch.no_grad():
        
        for test_idx, test_data in enumerate(test_loader):
            for key in test_data: test_data[key] = test_data[key].cuda()

            print("testing font %04d ..."%test_idx)

            dir_save = os.path.join(dir_res, "%04d"%test_idx)
            if not os.path.exists(dir_save):
                os.mkdir(dir_save)
                os.mkdir(os.path.join(dir_save, "imgs"))
                os.mkdir(os.path.join(dir_save, "svgs_single"))
                os.mkdir(os.path.join(dir_save, "svgs_merge"))
            svg_merge_dir = os.path.join(dir_save, "svgs_merge")

            iou_max = np.zeros(opts.char_num)
            idx_best_sample = np.zeros(opts.char_num)

            # syn_svg_merge_f = open(os.path.join(svg_merge_dir, f"{opts.name_ckpt}_syn_merge_{test_idx}_rand_{sample_idx}.html"), 'w') 
            syn_svg_merge_f = open(os.path.join(svg_merge_dir, f"{opts.name_ckpt}_syn_merge_{test_idx}.html"), 'w') 
    
            for sample_idx in range(opts.n_samples):
                ret_dict_test, loss_dict_test = model_main(test_data, mode='test')

                svg_sampled = ret_dict_test['svg']['sampled_1']
                sampled_svg_2 = ret_dict_test['svg']['sampled_2']

                img_trg = ret_dict_test['img']['trg']
                img_output = ret_dict_test['img']['out']
                trg_seq_gt = ret_dict_test['svg']['trg']

                img_sample_merge = torch.cat((img_trg.data, img_output.data), -2)
                save_file_merge = os.path.join(dir_save, "imgs", f"merge_{opts.img_size}.png")
                save_image(img_sample_merge, save_file_merge, nrow=8, normalize=True)    

                for char_idx in range(opts.char_num):
                    img_gt = (1.0 - img_trg[char_idx,...]).data
                    save_file_gt = os.path.join(dir_save,"imgs", f"{char_idx:02d}_gt.png")
                    save_image(img_gt, save_file_gt, normalize=True)

                    img_sample = (1.0 - img_output[char_idx,...]).data
                    save_file = os.path.join(dir_save,"imgs", f"{char_idx:02d}_{opts.img_size}.png")
                    save_image(img_sample, save_file, normalize=True)

                # write results w/o parallel refinement
                svg_dec_out = svg_sampled.clone().detach()
                for i, one_seq in enumerate(svg_dec_out):
                    syn_svg_outfile = os.path.join(os.path.join(dir_save, "svgs_single"), f"syn_{i:02d}_{sample_idx}_wo_refine.svg")

                    syn_svg_f_ = open(syn_svg_outfile, 'w')
                    try:
                        svg = render(one_seq.cpu().numpy())
                        syn_svg_f_.write(svg)
                        # syn_svg_merge_f.write(svg)
                        if i > 0 and i % 13 == 12:
                            syn_svg_f_.write('<br>')
                            # syn_svg_merge_f.write('<br>')
                        
                    except:
                        continue
                    syn_svg_f_.close()
                
                # write results w/ parallel refinement
                svg_dec_out = sampled_svg_2.clone().detach()
                for i, one_seq in enumerate(svg_dec_out):
                    syn_svg_outfile = os.path.join(os.path.join(dir_save, "svgs_single"), f"syn_{i:02d}_{sample_idx}_refined.svg")

                    syn_svg_f = open(syn_svg_outfile, 'w')
                    try:
                        svg = render(one_seq.cpu().numpy())
                        syn_svg_f.write(svg)
                        #syn_svg_merge_f.write(svg)
                        
                        #if i > 0 and i % 13 == 12:
                        #    syn_svg_merge_f.write('<br>')
                    except:
                        continue
                    syn_svg_f.close()
                    syn_img_outfile = syn_svg_outfile.replace('.svg', '.png')
                    svg2img(syn_svg_outfile, syn_img_outfile, img_size=opts.img_size)
                    iou_tmp, l1_tmp = cal_iou(syn_img_outfile, os.path.join(dir_save, "imgs", f"{i:02d}_{opts.img_size}.png"))
                    iou_tmp = iou_tmp
                    if iou_tmp > iou_max[i]:
                        iou_max[i] = iou_tmp
                        idx_best_sample[i] = sample_idx

            for i in range(opts.char_num):
                # print(idx_best_sample[i])
                syn_svg_outfile_best = os.path.join(os.path.join(dir_save, "svgs_single"), f"syn_{i:02d}_{int(idx_best_sample[i])}_refined.svg")
                syn_svg_merge_f.write(open(syn_svg_outfile_best, 'r').read())
                if i > 0 and i % 13 == 12:
                    syn_svg_merge_f.write('<br>')

            svg_target = trg_seq_gt.clone().detach()
            tgt_commands_onehot = F.one_hot(svg_target[:, :, :1].long(), 4).squeeze()
            tgt_args_denum = denumericalize(svg_target[:, :, 1:])
            svg_target = torch.cat([tgt_commands_onehot, tgt_args_denum], dim=-1)

            for i, one_gt_seq in enumerate(svg_target):
                # gt_svg_outfile = os.path.join(os.path.join(dir_save, "svgs_single"), f"gt_{i:02d}.svg")
                # gt_svg_f = open(gt_svg_outfile, 'w')
                gt_svg = render(one_gt_seq.cpu().numpy())
                # gt_svg_f.write(gt_svg)
                syn_svg_merge_f.write(gt_svg)
                # gt_svg_f.close()
                if i > 0 and i % 13 == 12:
                    syn_svg_merge_f.write('<br>')

            syn_svg_merge_f.close()

def main():
    
    opts = get_parser_main_model().parse_args()
    opts.name_exp = opts.name_exp + '_' + opts.model_name
    experiment_dir = os.path.join("./experiments", opts.name_exp)
    print(f"Testing on experiment {opts.name_exp}...")
    # Dump options
    test_main_model(opts)

if __name__ == "__main__":
    main()