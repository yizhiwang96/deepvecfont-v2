import argparse

def get_parser_main_model():
    parser = argparse.ArgumentParser()
    # basic parameters training related
    parser.add_argument('--model_name', type=str, default='main_model', choices=['main_model', 'neural_raster'], help='current model_name')
    parser.add_argument("--language", type=str, default='eng', choices=['eng', 'chn'])
    parser.add_argument('--bottleneck_bits', type=int, default=512, help='latent code number of bottleneck bits')
    parser.add_argument('--char_num', type=int, default=52, help='number of glyphs, original is 52')
    parser.add_argument('--ref_nshot', type=int, default=4, help='reference number')    
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=8, help='batch size when do validation')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--max_seq_len', type=int, default=51, help='maximum length of sequence')
    parser.add_argument('--dim_seq', type=int, default=12, help='the dim of each stroke in a sequence, 4 + 8, 4 is cmd, and 8 is args')
    parser.add_argument('--dim_seq_short', type=int, default=9, help='the short dim of each stroke in a sequence, 1 + 8, 1 is cmd class num, and 8 is args')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden_size')
    parser.add_argument('--dim_seq_latent', type=int, default=512, help='sequence encoder latent dim')
    parser.add_argument('--ngf', type=int, default=16, help='the basic num of channel in image encoder and decoder')
    parser.add_argument('--n_aux_pts', type=int, default=6, help='the number of aux pts in bezier curves for additional supervison')
    # experiment related
    parser.add_argument('--random_index', type=str, default='00')
    parser.add_argument('--name_ckpt', type=str, default='600_192921.ckpt')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs')
    parser.add_argument('--n_samples', type=int, default=20, help='the number of samples for each glyph when testing')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--ref_char_ids', type=str, default='0,1,26,27', help='default is A, B, a, b')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--name_exp', type=str, default='dvf')
    parser.add_argument('--data_root', type=str, default='./data/vecfont_dataset/')
    parser.add_argument('--freq_ckpt', type=int, default=50, help='save checkpoint frequency of epoch')
    parser.add_argument('--freq_sample', type=int, default=500, help='sample train output of steps')
    parser.add_argument('--freq_log', type=int, default=50, help='freq of showing logs')
    parser.add_argument('--freq_val', type=int, default=500, help='sample validate output of steps')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visulize loss')

    # loss weight
    parser.add_argument('--kl_beta', type=float, default=0.01, help='latent code kl loss beta')
    parser.add_argument('--loss_w_pt_c', type=float, default=0.001 * 10, help='the weight of perceptual content loss')
    parser.add_argument('--loss_w_l1', type=float, default=1.0 * 10, help='the weight of image reconstruction l1 loss')
    parser.add_argument('--loss_w_cmd', type=float, default=1.0, help='the weight of cmd loss')
    parser.add_argument('--loss_w_args', type=float, default=1.0, help='the weight of args loss')
    parser.add_argument('--loss_w_aux', type=float, default=0.01, help='the weight of pts aux loss')
    parser.add_argument('--loss_w_smt', type=float, default=10., help='the weight of smooth loss')
                            
    return parser
