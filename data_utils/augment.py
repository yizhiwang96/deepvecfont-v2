import argparse
import multiprocessing as mp
import os
import numpy as np
import math
import cairosvg
import shutil
from svg_utils import clockwise, render
from common_utils import affine_shear, affine_rotate, affine_scale, trans2_white_bg

def render_svg(svg_str, font_dir, char_idx, aug_idx, img_size):
    svg_html = render(svg_str)
    svg_path = open(f'{font_dir}/aug_svgs/{str(char_idx)}.svg', 'w')
    svg_path.write(svg_html)
    svg_path.close()
    cairosvg.svg2png(url=f'{font_dir}/aug_svgs/{str(char_idx)}.svg', 
                        write_to=f'{font_dir}/aug_imgs/{str(char_idx)}_{aug_idx}.png', output_width=img_size, output_height=img_size)
    img_arr = trans2_white_bg(f'{font_dir}/aug_imgs/{str(char_idx)}_{aug_idx}.png')
    return img_arr

def aug_rules(char_seq, aug_idx):
    if aug_idx == 0:
        return clockwise(affine_shear(char_seq, dx=0.2))['sequence']
    elif aug_idx == 1:
        return clockwise(affine_shear(char_seq, dy=-0.1))['sequence']
    elif aug_idx == 2:
        return clockwise(affine_scale(char_seq, 0.8))['sequence']
    elif aug_idx == 3:
        return clockwise(affine_rotate(char_seq, theta=5))['sequence']
    else:
        return clockwise(affine_rotate(char_seq, theta=-5))['sequence']

def copy_others(dir_src, dir_tgt):
    for item in ['class.npy', 'font_id.npy', 'seq_len.npy']:
        shutil.copy(f'{dir_src}/{item}', f'{dir_tgt}/{item}')

def apply_aug(opts):
    """
    applying data augmentation for Chinese fonts
    """
    data_path = os.path.join(opts.output_path, opts.language, opts.split)
    font_dirs_ = os.listdir(data_path)
    font_dirs = []
    for idx in range(len(font_dirs_)):
        if '_' not in font_dirs_[idx].split('/')[-1]:
            font_dirs.append(font_dirs_[idx])
    font_dirs.sort()
    num_fonts = len(font_dirs)
    print(f"Number {opts.split} fonts before processing", num_fonts)
    num_processes = mp.cpu_count() - 2
    fonts_per_process = num_fonts // num_processes + 1

    def process(process_id):
        for i in range(process_id * fonts_per_process, (process_id + 1) * fonts_per_process):
            if i >= num_fonts:
                break
            font_dir = os.path.join(data_path, font_dirs[i])
            font_seq = np.load(os.path.join(font_dir, 'sequence.npy')).reshape(opts.n_chars, opts.max_len, -1)

            ret_seq_list = []
            ret_img_list = []
            for k in range(opts.n_aug):
                os.makedirs(font_dir + '_' + str(k), exist_ok=True)
                ret_seq_list.append([])
                ret_img_list.append([])

            os.makedirs(f'{font_dir}/aug_svgs', exist_ok=True)
            os.makedirs(f'{font_dir}/aug_imgs', exist_ok=True)

            for j in range(opts.n_chars):
                char_seq = font_seq[j] # default as [71, 12]
                for k in range(opts.n_aug):
                    char_seq_aug = aug_rules(char_seq, k)
                    ret_seq_list[k].append(char_seq_aug)
                    img_arr = render_svg(char_seq_aug, font_dir, j, aug_idx=k, img_size=opts.img_size)
                    ret_img_list[k].append(img_arr)

            for k in range(opts.n_aug):
                ret_seq_list[k] = np.array(ret_seq_list[k]).reshape(opts.n_chars, opts.max_len * 10)
                ret_img_list[k] = np.array(ret_img_list[k]).reshape(opts.n_chars, opts.img_size, opts.img_size)
                np.save(os.path.join(font_dir + '_' + str(k), f'sequence.npy'), ret_seq_list[k])
                np.save(os.path.join(font_dir + '_' + str(k), f'rendered_{opts.img_size}.npy'), ret_img_list[k])
                copy_others(font_dir, font_dir + '_' + str(k))

    processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser(description="relax representation")
    parser.add_argument("--language", type=str, default='eng', choices=['eng', 'chn'])
    parser.add_argument("--output_path", type=str, default='../data/vecfont_dataset_/', help="Path to write the database to")
    parser.add_argument('--max_len', type=int, default=71, help="by default, 51 for english and 71 for chinese")
    parser.add_argument('--n_aug', type=int, default=5, help="for each font, augment it for n_aug times")
    parser.add_argument('--n_chars', type=int, default=52)
    parser.add_argument('--img_size', type=int, default=64, help="the height and width of glyph images")
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument('--debug', type=bool, default=True)
    opts = parser.parse_args()
    apply_aug(opts)

if __name__ == "__main__":
    main()




    