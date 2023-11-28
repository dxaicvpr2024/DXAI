import os

data_list = ['synth_white_rec', 'synth_color_rec', 'synth_rec_cir', 'Brats', 'DOTA_small-vehicle','DOTA_vehicle',
                'celeba_hq', 'xray', 'rambam_xray', 'rambam_clean', 'total_xray', 'synth_texture_lite', 'synth_grid_texture',
                'synth_grid_3dmns', 'afhq', 'dtd', 'dtd_zig_line_dot', 'histology_tumors', 'DOTA_plane', 'synth_color', 'synth_noise',
                'celeba_hq_noised_gray', 'afhq_noised_gray', 'synth_color_v2', 'apples', 'tomato', 'pepper']
data_name = data_list[14]
cuda_id = '0'
mission_name = 'try_dxai'

if not os.path.isdir('output_log'):
    os.makedirs('output_log')
            
log_file_name = 'output_log/log_'+data_name+'_'+mission_name+'.log'

os.system('echo cuda device id: '+cuda_id+' >> ' + log_file_name)

cmd = 'CUDA_VISIBLE_DEVICES='+cuda_id+' python main.py --mode eval\
      --sample_dir expr/samples_'+data_name+'_'+mission_name+'\
      --checkpoint_dir expr/checkpoints_'+data_name+'_'+mission_name+'\
      --src_dir assets/'+data_name+'\
      --ref_dir ../Data/'+data_name+'/val\
      --train_img_dir ../Data/'+data_name+'/train \
      --val_img_dir ../Data/'+data_name+'/val --resume_iter 0\
      --data_name ' + data_name + '\
      --mission_name ' + mission_name + '\
      --use_pretrained_classifier 1 --classifier_type resnet18\
      --num_branches 5\
      --img_channels 3 --branch_dimin 7 --softmax_temp 1\
      --img_size 256 --batch_size 2\
      --sample_every 50 --save_every 50 --total_iters 51 \
      --max_eval_iter 15\
      --lr 0.0001  --f_lr 0.000001\
      --noise_every 500000  &> '+log_file_name+' &'

print(cmd)
os.system(cmd)
print('end of training')
#os.system(cmd.replace('--mode train', '--mode eval'))
