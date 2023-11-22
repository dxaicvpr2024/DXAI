import argparse
def load_args():
    parser = argparse.ArgumentParser()
    # changed
    base_directory = 'data/xray_data/'  # synth_data2/'  #
    base_sample_directory = 'assets/representative/xray_data/'  # synth_data2/'   #
    parser.add_argument('--train_img_dir', type=str, default=base_directory + 'train', help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default=base_directory + 'val', help='Directory containing validation images')
    parser.add_argument('--lambda_reg', type=float, default=1, help='Weight for R1 regularization')
    parser.add_argument('--lambda_adv', type=float, default=2, help='adversarial loss for generator')
    parser.add_argument('--lambda_class_fake', type=float, default=2, help='adversarial loss for generator')

    parser.add_argument('--lambda_sim', type=float, default=4, help='Weight for similarity loss')
    parser.add_argument('--lambda_grad_sim', type=float, default=4, help='Weight for gradients similarity loss')
    parser.add_argument('--lambda_ano_sim', type=float, default=4, help='Weight for anomaly similarity loss')

    parser.add_argument('--lambda_params_corr', type=float, default=1, help='Weight for params corr regularization')

    parser.add_argument('--softmax_temp', type=float, default=1, help='Weight for params corr regularization')

    parser.add_argument('--zero_st', type=int, default=1, help='')
    parser.add_argument('--data_range_norm', type=int, default=1)
    parser.add_argument('--out_features', type=int, default=1, help='')

    parser.add_argument('--use_pretrained_classifier', type=int, default=0, help='if use given classifier for xai')
    parser.add_argument('--classifier_type', type=str, default='', help='stargan or resnet18 or vgg16 or classifier2')
    parser.add_argument('--classifier_ckpt_path', type=str, default='./', help='checkpoints to classifier weights')
    
    parser.add_argument('--test_number', type=int, default=1, help='test bigger than 1 for collect statistic')
    
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--img_channels', type=int, default=3, help='Image channels')
    parser.add_argument('--alpha_blend', type=int, default=1)
    parser.add_argument('--noise_every', type=int, default=2e3, help='Image resolution')
    # new
    parser.add_argument('--num_branches', type=int, default=6)
    parser.add_argument('--branch_dimin', type=int, default=7)  # generator's dimin equals num_branches * branch_dimin (63=9*7 by default)
    parser.add_argument('--use_classifier', type=bool, default=False)
    parser.add_argument('--data_name', type=str, default=' ')
    parser.add_argument('--mission_name', type=str, default='')

    parser.add_argument('--style_per_block', type=int, default=True, help='indicates how and whether style code (of a specific image) is should be mixed upon testing')

    # original stargan-v2
    parser.add_argument('--mode', default='train', help='train or eval')
    parser.add_argument('--num_AdaInResBlks', type=int, default=1, help='number of AdaIn blocks, which will be assigned as the number of style vectors needed for inference')

    # model arguments
    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')

    parser.add_argument('--w_hpf', type=float, default=0, help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--total_iters', type=int, default=1000000, help='Number of total iterations')
   
    parser.add_argument('--max_eval_iter', type=int, default=1500, help='Number of total iterations')
        
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=30,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default=base_sample_directory + 'src', help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default=base_sample_directory + 'ref', help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=3000)  # 10000
    parser.add_argument('--save_every', type=int, default=10000)  # 10000 50000
    parser.add_argument('--eval_every', type=int, default=50000e2)

    args = parser.parse_args()
    if args.num_branches == 1:
        args.alpha_blend = False
    return args