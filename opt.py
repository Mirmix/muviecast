import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='./data/',
                        help='root directory of dtu dataset')
    parser.add_argument('--img_wh', nargs=2, type=int, default=[640, 352],
                        help='Width and height of the image, separated by a comma')
    parser.add_argument('--scan_name', type=str, default='Train',
                        help='which scan to train/val')
    parser.add_argument('--img_ext', type=str, default='jpg',
                        help='image extension')
    parser.add_argument('--output_dir', type=str, default='images',
                        help='output folder')
    parser.add_argument('--n_views', type=int, default=3,
                        help='number of views (including ref) to be used in training')
    parser.add_argument('--levels', type=int, default=3, choices=[3],
                        help='number of FPN levels (fixed to be 3!)')
    parser.add_argument('--depth_interval', type=float, default=2.65,
                        help='depth interval for the finest level, unit in mm')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8, 32, 48],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0, 2.0, 4.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--num_groups', type=int, default=8, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8')
    parser.add_argument('--loss_type', type=str, default='sl1',
                        choices=['sl1'],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')

    parser.add_argument('--num_gpus', type=int, default=2,
                        help='number of gpus')

    parser.add_argument('--checkpoint_interval', type=int, default=200,
                        help='interval to save checkpoints')
    parser.add_argument('--sample_interval', type=int, default=200,
                        help='interval to save samples')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--patchmatch_weights', type=str, default='patchmatchnet_params.ckpt',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--casmvs_weights', type=str, default='casmvsnet_g8.ckpt',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--adain_weights', type=str, default='adain.tar',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--unet_weights', type=str, default='gogh.pth',
                        help='pretrained checkpoint path to load')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='multisteplr',
                        help='scheduler type',
                        choices=['multisteplr', 'steplr', 'cosine', 'poly'])

    # PatchMatchNet module options (only used when not loading from file)
    parser.add_argument("--patchmatch_interval_scale", nargs="+", type=float, default=[0.005, 0.0125, 0.025],
                        help="normalized interval in inverse depth range to generate samples in local perturbation")
    parser.add_argument("--patchmatch_range", nargs="+", type=int, default=[6, 4, 2],
                        help="fixed offset of sampling points for propagation of patch match on stages 1,2,3")
    parser.add_argument("--patchmatch_iteration", nargs="+", type=int, default=[1, 2, 2],
                        help="num of iteration of patch match on stages 1,2,3")
    parser.add_argument("--patchmatch_num_sample", nargs="+", type=int, default=[8, 8, 16],
                        help="num of generated samples in local perturbation on stages 1,2,3")
    parser.add_argument("--propagate_neighbors", nargs="+", type=int, default=[0, 8, 16],
                        help="num of neighbors for adaptive propagation on stages 1,2,3")
    parser.add_argument("--evaluate_neighbors", nargs="+", type=int, default=[9, 9, 9],
                        help="num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3")

    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####d
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################
    #### params for loss ####
    parser.add_argument('--lambda_content', type=float, default=1e3,
                        help='weight for content loss')
    parser.add_argument('--lambda_style', type=float, default=1e8,
                        help='weight for style loss')
    parser.add_argument('--lambda_volume', type=float, default=0,
                        help='weight for volume loss')
    parser.add_argument('--lambda_depth', type=float, default=2e3,
                        help='weight for depth loss')
    parser.add_argument('--lambda_structure', type=float, default=2e4,
                        help='weight for image structure loss')
    parser.add_argument('--lambda_nnfm', type=float, default=0,
                        help='weight for nnfm loss')
    parser.add_argument('--volume_loss_type', type=str, default='smoothl1',
                        help='volume_loss_type',
                        choices=['smoothl1', 'mse', 'kl', 'custom'])
    ###########################
    #### params for style ####
    parser.add_argument('--style_dir', type=str, default='./data/styles/',
                        help='style image path')
    parser.add_argument('--style_name', type=str, default='gogh',
                        help='style image name')

    ###########################
    parser.add_argument('--use_amp', default=False, action="store_true",
                        help='use mixed precision training (NOT SUPPORTED!)')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    parser.add_argument('--use_casmvsnet', default=False, action="store_true",
                        help='use_casmvsnet')
    parser.add_argument('--use_adain', default=False, action="store_true",
                        help='use adaIN')
    parser.add_argument('--color_adjust', default=False, action="store_true",
                        help='use color adjust')
    parser.add_argument('--ablation_suffix', type=str, default='',
                        help='style image name')

    return parser.parse_args()
