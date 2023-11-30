import os

data_list = ['celeba_hq', 'afhq']
data_name = data_list[0]
cuda_id = '0'
mission_name = 'try_dxai'

if not os.path.isdir('output_log'):
    os.makedirs('output_log')
            
log_file_name = 'output_log/log_'+data_name+'_'+mission_name+'.log'

os.system('echo cuda device id: '+cuda_id+' >> ' + log_file_name)

cmd = 'CUDA_VISIBLE_DEVICES='+cuda_id+' python main.py --mode train\
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
      --img_channels 3\
      --img_size 256 --batch_size 2\
      --sample_every 150 --save_every 150 --total_iters 151 \
      --max_eval_iter 15 &> '+log_file_name+' &'

print(cmd)
os.system(cmd)
print('end of training')
#os.system(cmd.replace('--mode train', '--mode eval'))
