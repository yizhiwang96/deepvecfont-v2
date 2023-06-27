# data loader for training main model
import os
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as T
import sys
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')


class SVGDataset(data.Dataset):
    def __init__(self, root_path, img_size=128, lang='eng', char_num=52, max_seq_len=51, dim_seq=10,  transform=None, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        self.char_num = char_num
        self.max_seq_len = max_seq_len
        self.dim_seq = dim_seq
        self.trans = transform
        self.font_paths = []
        self.dir_path = os.path.join(root_path, lang, self.mode)
        for root, dirs, files in os.walk(self.dir_path):
            depth = root.count('/') - self.dir_path.count('/')
            if depth == 0:
                for dir_name in dirs:
                    self.font_paths.append(os.path.join(self.dir_path, dir_name))
        self.font_paths.sort()
        print(f"Finished loading {mode} paths, number: {str(len(self.font_paths))}")
        
    def __getitem__(self, index):
        item = {}
        font_path = self.font_paths[index]
        item = {}
        item['class'] = torch.LongTensor(np.load(os.path.join(font_path, 'class.npy')))
        item['seq_len'] = torch.LongTensor(np.load(os.path.join(font_path, 'seq_len.npy')))
        item['sequence'] = torch.FloatTensor(np.load(os.path.join(font_path, 'sequence_relaxed.npy'))).view(self.char_num, self.max_seq_len, self.dim_seq)
        item['pts_aux'] = torch.FloatTensor(np.load(os.path.join(font_path, 'pts_aux.npy')))
        item['rendered'] = torch.FloatTensor(np.load(os.path.join(font_path, 'rendered_' + str(self.img_size) + '.npy'))).view(self.char_num, self.img_size, self.img_size) / 255.
        item['rendered'] = self.trans(item['rendered'])
        item['font_id'] = torch.FloatTensor(np.load(os.path.join(font_path, 'font_id.npy')).astype(np.float32))
        return item

    def __len__(self):
        return len(self.font_paths)


def get_loader(root_path, img_size, lang, char_num, max_seq_len, dim_seq, batch_size, mode='train'):
    SetRange = T.Lambda(lambda X: 1. - X )  # convert [0, 1] -> [0, 1]
    transform = T.Compose([SetRange])
    dataset = SVGDataset(root_path, img_size, lang, char_num, max_seq_len, dim_seq, transform, mode)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=batch_size)
    return dataloader

if __name__ == '__main__':
    root_path = 'data/new_data'
    max_seq_len = 51
    dim_seq = 10
    batch_size = 1
    char_num = 52

    loader = get_loader(root_path, char_num, max_seq_len, dim_seq, batch_size, 'train')
    fout = open('train_id_record_old.txt','w')
    for idx, batch in enumerate(loader):
        binary_fp = batch['font_id'].numpy()[0][0]
        fout.write("%05d"%int(binary_fp) + '\n')

