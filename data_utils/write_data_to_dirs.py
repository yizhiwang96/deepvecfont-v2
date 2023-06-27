import argparse
import multiprocessing as mp
import os
import pickle
import numpy as np
import svg_utils

def exist_empty_imgs(imgs_array, num_chars):
    for char_id in range(num_chars):
        print(np.max(imgs_array[char_id]))
        input()
        if np.max(imgs_array[char_id]) == 0:
            return True
    return False

def create_db(opts, output_path, log_path):
    charset = open(f"../data/char_set/{opts.language}.txt", 'r').read()
    print("Process sfd to npy files in dirs....")
    sdf_path = os.path.join(opts.sfd_path, opts.language, opts.split)
    all_font_ids = sorted(os.listdir(sdf_path))
    num_fonts = len(all_font_ids)
    num_fonts_w = len(str(num_fonts))
    print(f"Number {opts.split} fonts before processing", num_fonts)
    num_processes = mp.cpu_count() - 2
    fonts_per_process = num_fonts // num_processes + 1
    num_chars = len(charset)
    num_chars_w = len(str(num_chars))


    def process(process_id):

        cur_process_log_file = open(os.path.join(log_path, f'log_{opts.split}_{process_id}.txt'), 'w')
        for i in range(process_id * fonts_per_process, (process_id + 1) * fonts_per_process):
            if i >= num_fonts:
                break
            font_id = all_font_ids[i]
            cur_font_sfd_dir = os.path.join(sdf_path, font_id)
            cur_font_glyphs = []

            if not os.path.exists(os.path.join(cur_font_sfd_dir, 'imgs_' + str(opts.img_size) + '.npy')):
                continue
            
            # a whole font as an entry
            for char_id in range(num_chars):
                print(char_id)
                if not os.path.exists(os.path.join(cur_font_sfd_dir, '{}_{num:0{width}}.sfd'.format(font_id, num=char_id, width=num_chars_w))):
                    break

                char_desp_f = open(os.path.join(cur_font_sfd_dir, '{}_{num:0{width}}.txt'.format(font_id, num=char_id, width=num_chars_w)), 'r')
                char_desp = char_desp_f.readlines()
                sfd_f = open(os.path.join(cur_font_sfd_dir, '{}_{num:0{width}}.sfd'.format(font_id, num=char_id, width=num_chars_w)), 'r')
                sfd = sfd_f.read()

                uni = int(char_desp[0].strip())
                width = int(char_desp[1].strip())
                vwidth = int(char_desp[2].strip())
                char_idx = char_desp[3].strip()
                font_idx = char_desp[4].strip()

                cur_glyph = {}
                cur_glyph['uni'] = uni
                cur_glyph['width'] = width
                cur_glyph['vwidth'] = vwidth
                cur_glyph['sfd'] = sfd
                cur_glyph['id'] = char_idx
                cur_glyph['binary_fp'] = font_idx

                if not svg_utils.is_valid_glyph(cur_glyph):
                    msg = f"font {font_idx}, char {char_idx} is not a valid glyph\n"
                    cur_process_log_file.write(msg)
                    char_desp_f.close()
                    sfd_f.close()
                    # use the font whose all glyphs are valid
                    break
                pathunibfp = svg_utils.convert_to_path(cur_glyph)

                if not svg_utils.is_valid_path(pathunibfp):
                    msg = f"font {font_idx}, char {char_idx}'s sfd is not a valid path\n"
                    cur_process_log_file.write(msg)
                    char_desp_f.close()
                    sfd_f.close()
                    break

                example = svg_utils.create_example(pathunibfp)

                cur_font_glyphs.append(example)
                char_desp_f.close()
                sfd_f.close()
            
            if len(cur_font_glyphs) == num_chars:
                # use the font whose all glyphs are valid
                # merge the whole font

                rendered = np.load(os.path.join(cur_font_sfd_dir, 'imgs_' + str(opts.img_size) + '.npy'))

                if (rendered[0] == rendered[1]).all() == True:
                    continue
 
                sequence = []
                seq_len = []
                binaryfp = []
                char_class = []
                for char_id in range(num_chars):
                    example = cur_font_glyphs[char_id]
                    sequence.append(example['sequence'])
                    seq_len.append(example['seq_len'])
                    char_class.append(example['class'])
                    binaryfp = example['binary_fp']
                if not os.path.exists(os.path.join(output_path, '{num:0{width}}'.format(num=i, width=num_fonts_w))):
                    os.mkdir(os.path.join(output_path, '{num:0{width}}'.format(num=i, width=num_fonts_w)))

                np.save(os.path.join(output_path, '{num:0{width}}'.format(num=i, width=num_fonts_w), 'sequence.npy'), np.array(sequence))
                np.save(os.path.join(output_path, '{num:0{width}}'.format(num=i, width=num_fonts_w), 'seq_len.npy'), np.array(seq_len))
                np.save(os.path.join(output_path, '{num:0{width}}'.format(num=i, width=num_fonts_w), 'class.npy'), np.array(char_class))
                np.save(os.path.join(output_path, '{num:0{width}}'.format(num=i, width=num_fonts_w), 'font_id.npy'), np.array(binaryfp))
                np.save(os.path.join(output_path, '{num:0{width}}'.format(num=i, width=num_fonts_w), 'rendered_' + str(opts.img_size) + '.npy'), rendered)

    processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("Finished processing all sfd files, logs (invalid glyphs and paths) are saved to", log_path)


def cal_mean_stddev(opts, output_path):
    print("Calculating all glyphs' mean stddev ....")
    charset = open(f"../data/char_set/{opts.language}.txt", 'r').read()
    font_paths = []
    for root, dirs, files in os.walk(output_path):
        for dir_name in dirs:
            font_paths.append(os.path.join(output_path, dir_name))
    font_paths.sort()
    num_fonts = len(font_paths)
    num_processes = mp.cpu_count() - 2
    fonts_per_process = num_fonts // num_processes + 1
    num_chars = len(charset) 
    manager = mp.Manager()
    return_dict = manager.dict()
    main_stddev_accum = svg_utils.MeanStddev()

    def process(process_id, return_dict):
        mean_stddev_accum = svg_utils.MeanStddev()
        cur_sum_count = mean_stddev_accum.create_accumulator()
        for i in range(process_id * fonts_per_process, (process_id + 1) * fonts_per_process):
            if i >= num_fonts:
                break
            cur_font_path = font_paths[i]
            for charid in range(num_chars):
                cur_font_char = {}
                cur_font_char['seq_len'] = np.load(os.path.join(cur_font_path, 'seq_len.npy')).tolist()[charid]
                cur_font_char['sequence'] = np.load(os.path.join(cur_font_path, 'sequence.npy')).tolist()[charid]
                cur_sum_count = mean_stddev_accum.add_input(cur_sum_count, cur_font_char)
        return_dict[process_id] = cur_sum_count
    processes = [mp.Process(target=process, args=[pid, return_dict]) for pid in range(num_processes)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    merged_sum_count = main_stddev_accum.merge_accumulators(return_dict.values())
    output = main_stddev_accum.extract_output(merged_sum_count)
    mean = output['mean']
    stdev = output['stddev']
    mean = np.concatenate((np.zeros([4]), mean[4:]), axis=0)
    stdev = np.concatenate((np.ones([4]), stdev[4:]), axis=0)
    # finally, save the mean and stddev files
    output_path_ = os.path.join(opts.output_path, opts.language)
    np.save(os.path.join(output_path_, 'mean'), mean)
    np.save(os.path.join(output_path_, 'stdev'), stdev)

    # rename npy to npz, don't mind about it, just some legacy issue 
    os.rename(os.path.join(output_path_, 'mean.npy'), os.path.join(output_path_, 'mean.npz'))
    os.rename(os.path.join(output_path_, 'stdev.npy'), os.path.join(output_path_, 'stdev.npz'))

def main():
    parser = argparse.ArgumentParser(description="LMDB creation")
    parser.add_argument("--language", type=str, default='eng', choices=['eng', 'chn'])
    parser.add_argument("--ttf_path", type=str, default='../data/font_ttfs')
    parser.add_argument('--sfd_path', type=str, default='../data/font_sfds')
    parser.add_argument("--output_path", type=str, default='../data/vecfont_dataset_/', help="Path to write the database to")
    parser.add_argument('--img_size', type=int, default=64, help="the height and width of glyph images")
    parser.add_argument("--split", type=str, default='train')
    # parser.add_argument("--log_dir", type=str, default='../data/font_sfds/log/')
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="0 all, 1 create db, 2 cal stddev")

    opts = parser.parse_args()
    assert os.path.exists(opts.sfd_path), "specified sfd glyphs path does not exist"

    output_path = os.path.join(opts.output_path, opts.language, opts.split)
    log_path = os.path.join(opts.sfd_path, opts.language, 'log')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opts.phase <= 1:
        create_db(opts, output_path, log_path)

    if opts.phase <= 2 and opts.split == 'train':
        cal_mean_stddev(opts, output_path)

    
if __name__ == "__main__":
    main()