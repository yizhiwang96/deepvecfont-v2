import argparse
import multiprocessing as mp
import os
import numpy as np

def numericalize(cmd, n=64):
    """NOTE: shall only be called after normalization"""
    cmd = ((cmd) / 30 * n).round().clip(min=0, max=n-1).astype(int)
    return cmd

def denumericalize(cmd, n=64):
    cmd = cmd / n * 30
    return cmd

def cal_aux_bezier_pts(font_seq, opts):
    """
    calculate aux pts along bezier curves
    """
    pts_aux_all = []

    for j in range(opts.n_chars):
        char_seq = font_seq[j] # shape: opts.max_len ,12
        pts_aux_char = []
        for k in range(opts.max_len):
            stroke_seq = char_seq[k]
            stroke_cmd = np.argmax(stroke_seq[:4], -1)
            stroke_seq[4:] = denumericalize(numericalize(stroke_seq[4:]))
            p0, p1, p2, p3 = stroke_seq[4:6], stroke_seq[6:8], stroke_seq[8:10], stroke_seq[10:12]
            pts_aux_stroke = []
            if stroke_cmd == 0:
                for t in range(6):
                    pts_aux_stroke.append(0)
            elif stroke_cmd == 1: # move
                for t in [0.25, 0.5, 0.75]:
                    coord_t = p0 + t*(p3-p0)
                    pts_aux_stroke.append(coord_t[0])
                    pts_aux_stroke.append(coord_t[1])
            elif stroke_cmd == 2: # line
                for t in [0.25, 0.5, 0.75]:
                    coord_t = p0 + t*(p3-p0)
                    pts_aux_stroke.append(coord_t[0])
                    pts_aux_stroke.append(coord_t[1])
            elif stroke_cmd == 3: # curve
                for t in [0.25, 0.5, 0.75]:
                    coord_t = (1-t)*(1-t)*(1-t)*p0 + 3*t*(1-t)*(1-t)*p1 + 3*t*t*(1-t)*p2 + t*t*t*p3
                    pts_aux_stroke.append(coord_t[0])
                    pts_aux_stroke.append(coord_t[1])
            
            pts_aux_stroke = np.array(pts_aux_stroke)
            pts_aux_char.append(pts_aux_stroke)
            
        pts_aux_char = np.array(pts_aux_char)
        pts_aux_all.append(pts_aux_char)
    
    pts_aux_all = np.array(pts_aux_all)

    return pts_aux_all


def relax_rep(opts):
    """
    relaxing the sequence representation, details are shown in paper
    """
    data_path = os.path.join(opts.output_path, opts.language, opts.split)
    font_dirs = os.listdir(data_path)
    font_dirs.sort()
    num_fonts = len(font_dirs)
    print(f"Number {opts.split} fonts before processing", num_fonts)
    num_processes = mp.cpu_count() - 2
    # num_processes = 1
    fonts_per_process = num_fonts // num_processes + 1

    def process(process_id):

        for i in range(process_id * fonts_per_process, (process_id + 1) * fonts_per_process):
            if i >= num_fonts:
                break

            font_dir = os.path.join(data_path, font_dirs[i])
            font_seq = np.load(os.path.join(font_dir, 'sequence.npy')).reshape(opts.n_chars, opts.max_len, -1)
            font_len = np.load(os.path.join(font_dir, 'seq_len.npy')).reshape(-1)
            cmd = font_seq[:, :, :4]
            args = font_seq[:, :, 4:]

            ret = []
            for j in range(opts.n_chars):
            
                char_cmds = cmd[j]
                char_args = args[j]
                char_len = font_len[j]
                new_args = []
                for k in range(char_len):
                    cur_cls = np.argmax(char_cmds[k], -1)
                    cur_arg = char_args[k]
                    if k - 1 > -1:
                        pre_arg = char_args[k - 1]
                    if cur_cls == 1: # when k == 0, cur_cls == 1
                        cur_arg = np.concatenate((np.array([cur_arg[-2], cur_arg[-1]]), cur_arg), -1)
                    else:
                        cur_arg = np.concatenate((np.array([pre_arg[-2], pre_arg[-1]]), cur_arg), -1)
                    new_args.append(cur_arg)
                
                while(len(new_args)) < opts.max_len:
                    new_args.append(np.array([0, 0, 0, 0, 0, 0, 0, 0]))

                new_args = np.array(new_args)
                new_seq = np.concatenate((char_cmds, new_args),-1)
                ret.append(new_seq)
            ret = np.array(ret)
            # write relaxed version of sequence.npy
            np.save(os.path.join(font_dir, 'sequence_relaxed.npy'), ret.reshape(opts.n_chars, -1))

            pts_aux = cal_aux_bezier_pts(ret, opts)
            np.save(os.path.join(font_dir, 'pts_aux.npy'), pts_aux)

    processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
def main():
    parser = argparse.ArgumentParser(description="relax representation")
    parser.add_argument("--language", type=str, default='eng', choices=['eng', 'chn'])
    parser.add_argument("--output_path", type=str, default='../data/vecfont_dataset_/', help="Path to write the database to")
    parser.add_argument('--max_len', type=int, default=51, help="by default, 51 for english and 71 for chinese")
    parser.add_argument('--n_chars', type=int, default=52)
    parser.add_argument("--split", type=str, default='train')
    opts = parser.parse_args()
    relax_rep(opts)

if __name__ == "__main__":
    main()