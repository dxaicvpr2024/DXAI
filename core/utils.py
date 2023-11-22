"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from os.path import join as ospj
import json
from tqdm import tqdm
import ffmpeg
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torchmetrics
from core.branch_utils import sum_groups
import cv2


def my_gradient(x):
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2], 1)).to(x.device)
    dx = torch.cat((z, dx), dim=3)
    dy = torch.cat((z.permute((0, 1, 3, 2)), dy), dim=2)
    return dx, dy

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename, denorm=True):
    if denorm:
        x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)



@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    # masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat


def mix_2_styles(s_src, s_ref, mode, style_dim):

    if mode == 'interp':
        s_combined = 0.5 * (s_src + s_ref)
    elif mode == 'twopart':
        start = 0 * style_dim  # start and end has to be in kfulot of style_dim, since the s's are colusracked style vectors
        end = 3 * style_dim

        s_combined = torch.cat((s_ref[:, :start],s_src[:, start:end], s_ref[:, end:]), dim=1)

        # print(s_combined)
        # print(s_src)
        # print(s_ref)

        # print(s_ref.shape)
        # print(s_src.shape)
        # print(s_ref[:, :start].shape)
        # print(s_src[:, start:end].shape)
        # print(s_ref[:, end:].shape)
        # print(s_src.shape)

    return s_combined, s_src, s_ref


@torch.no_grad()
def test_psi(nets, args, x_src, y_src, x_ref, y_ref, filename, use_z=False, time_factor=1, return_psi = False, alpha=1, phi_filt=None, psi_filt=None, filt_len=0, dont_cat_ref=True, step=0, batch_idx=0):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)

    y_ref = torch.tensor(list(range(min(args.val_batch_size, args.num_domains)))).to(x_src.device)

    z_trg = torch.randn(y_ref.size(0), args.latent_dim).to(x_src.device)
    s_ref = nets.mapping_network(z_trg, y_ref)

    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    print(len(s_ref_list))


    for i, s_ref in enumerate(s_ref_list):
        print(i)
        x_fake, Psi, Phi, Residual, Masks = nets.generator(x_src, s_ref)
        if args.zero_st:
            Psi[y_src != y_ref[i], 0:args.img_channels, :, :] = 0*Psi[y_src != y_ref[i], 0:args.img_channels, :, :]
            x_fake = sum_groups(Psi, phi_depth=args.img_channels)

        x_concat = [x_src]
        mask_concat = [x_src]
        x_cumsum_concat = [x_src]

        # concatenate as list
        if dont_cat_ref:
            x_fake_with_ref = x_fake
        else:
            x_fake_with_ref = torch.cat([x_ref[i:i + 1], x_fake], dim=0)  # option 2 (orig): concat with reference im

        x_concat += [x_fake_with_ref]
        mask_concat += [x_fake_with_ref]

        accum = Residual

        ## show psis

        assert int(Psi.shape[1]/args.img_channels) == Psi.shape[1]/args.img_channels

        for p in range(int(Psi.shape[1]/args.img_channels)):
            # choose psi (each psi is 3-channel, so that their sum is RGB)
            psi = Psi[:, args.img_channels*p:(args.img_channels*p+args.img_channels), :, :]
            mask = Masks[:, args.img_channels*p:(args.img_channels*p+args.img_channels), :, :]
            accum += psi

            # concatenate as list
            psi_with_ref = psi #torch.cat([wb, psi], dim=0)  # option 1 (as above)
            mask_with_ref = mask #torch.cat([wb, mask], dim=0)  # option 1 (as above)
            psi_cumsum_with_ref = accum # torch.cat([wb, accum], dim=0)

            x_concat += [psi_with_ref]
            mask_concat += [mask_with_ref]
            x_cumsum_concat += [psi_cumsum_with_ref]

        # convert concatenated list to pytorch tensor
        x_concat = torch.cat(x_concat, dim=0)

        # save
        ref_label = str(int(y_ref[i].detach().cpu()))
        fname = filename+'_ref_'+ref_label + '_' + str(i) + '.png'
        if batch_idx > 0:
            fname = filename+'_ref_'+ref_label + '_' + str(i)  + '_' + str(batch_idx)+ '.png'
        print('saving ' + fname)
        print(x_concat.shape)
        save_image(x_concat, N, fname)


@torch.no_grad()
def test_anomaly(nets, args, x_src, y_src, x_ref, filename, use_z=True, batch_idx=0, mean_num=1):

    N, C, H, W = x_src.size()

    x_trg = torch.zeros((N, C, H, W, args.num_domains)).to(x_src.device)
    psi_trg = torch.zeros((N, args.num_branches*C, H, W, args.num_domains)).to(x_src.device)
    y_list = list(range(min(args.val_batch_size, args.num_domains)))
    for mean_i in range(mean_num):
        z_trg = torch.randn(x_ref.size(0), args.latent_dim).to(x_ref.device)
        for y in y_list:
            #print('y: ', y)
            y_ref = y*torch.ones(1).to(x_src.device).long()
            #print(y_ref)
            if use_z:
                s_ref = nets.mapping_network(z_trg, y_ref)
            else:
                s_ref = nets.style_encoder(x_src, y_ref)
            s_ref = s_ref.repeat((x_src.size(0), 1))
            x_fake, Psi, Phi, Residual, Masks = nets.generator(x_src, s_ref)
            if args.zero_st:
                Psi[y_src != y_ref, 0:args.img_channels, :, :] = 0*Psi[y_src != y_ref, 0:args.img_channels, :, :]
                x_fake = sum_groups(Psi, phi_depth=args.img_channels)
            x_trg[:, :, :, :, y] += x_fake / mean_num
            psi_trg[:, :, :, :, y] += Psi / mean_num

    x_src_expand = x_src[:,:,:,:,None]
    err = (x_src_expand - x_trg) #*torch.abs(psi_trg[:, 0:args.img_channels, :, :, 0]-psi_trg[:, 0:args.img_channels, :, :, 1])
    # print(err.shape)
    mean_err = err.mean(dim=(1,2,3))
    # print(mean_err.shape)
    _, y_est = torch.min(mean_err, dim=1)
    #print(y_est)
    x_rec = torch.zeros_like(x_src).to(x_src.device)
    x_real_rec = torch.zeros_like(x_src).to(x_src.device)
    anomaly_diff_clean = torch.zeros_like(x_src).to(x_src.device)

    x_anomaly = torch.zeros_like(x_src).to(x_src.device)
    err2show = torch.zeros_like(x_src).to(x_src.device)
    for i in range(x_src.size(0)):
        x_rec[i] = x_trg[i, :, :, :, y_est[i]].squeeze(-1)
        x_real_rec[i] = x_trg[i, :, :, :, y_src[i]].squeeze(-1)
        x_anomaly[i] = psi_trg[i, 0:args.img_channels, :, :, y_src[i]].squeeze(-1)
        err2show[i] = err[i, :, :, :, y_src[i]].squeeze(-1)
        anomaly_diff_clean[i] = (psi_trg[i, 0:args.img_channels, :, :, y_est[i]]-psi_trg[i, 0:args.img_channels, :, :, 1-y_est[i]]).abs()
        anomaly_diff_clean[i] = (anomaly_diff_clean[i]*(psi_trg[i, 0:args.img_channels, :, :, y_src[i]].abs())).squeeze(-1)
        

    if batch_idx<20:
        x2show = [x_src]
        x2show += [x_real_rec]
        x2show += [tensor_contrast_stretch(err2show)]
        for dd in range(args.num_domains):
            x2show += [psi_trg[:, 0:args.img_channels, :, :, dd].squeeze(-1)]
        
        if args.num_domains==2:
            anomaly_diff = x_trg[:, :, :, :, 0]-x_trg[:, :, :, :, 1]
            x2show += [(anomaly_diff)]
        
            gx_0, gy_0 = my_gradient(x_trg[:, :, :, :, 0])
            TV_0 = gx_0.abs() + gy_0.abs()
            gx_1, gy_1 = my_gradient(x_trg[:, :, :, :, 1])
            TV_1 = gx_1.abs() + gy_1.abs()
            TV_diff = TV_0 - TV_1
            
            if args.img_channels == 1:
                anomaly_diff_abs = anomaly_diff.abs() 
                TV_diff_abs = TV_diff.abs()
            else:
                anomaly_diff_abs = anomaly_diff.abs().mean(1, keepdim=True).repeat(1, 3, 1, 1)
                TV_diff_abs = TV_diff.abs().mean(1, keepdim=True).repeat(1, 3, 1, 1)
        
            x2show += [2 * anomaly_diff_abs - 1]
            x2show += [2*tensor_contrast_stretch(anomaly_diff_clean**0.25)-1]
        
        #blur_TV = gradual_gaussian_blur(TV_diff_abs, kernel_size=13, sigma=3, filter_num=5, channels=args.img_channels)
        
        x2show += [make_anomaly_heatmap(x_anomaly, x_src)]
        #anomaly_heatmap[blur_TV < 0.02] = 0
        #x2show += [blur_TV]
        #x2show += [2 * anomaly_heatmap*(x_trg[:, :, :, :, 0]*x_trg[:, :, :, :, 1]).abs().sqrt() - 1]
        
        #if mean_num > 1:
        #    x2show += [x_trg[:, :, :, :, 0].squeeze(-1)]
        #    x2show += [x_trg[:, :, :, :, 1].squeeze(-1)]
        
        if args.img_channels == 1:
            for ii in range(len(x2show)):
                if x2show[ii].shape[1]==1:
                    x2show[ii] = torch.cat((x2show[ii],x2show[ii],x2show[ii]), dim=1)
        
        x2show = torch.cat(x2show, dim=0)
        # print(x2show.shape)
        # save
        fname = filename + '_anomalies' + '_' + str(batch_idx)+'_mean_num_'+str(mean_num)+'_.png'
        print('saving anomaly ' + fname)
        save_image(x2show, N, fname)
    SNR = 20*torch.log10(torch.sqrt((x_src**2).mean()/(((x_src-x_rec)**2).mean()+1e-8)))

    # M, _ = x_src.reshape((N, C*H*W)).max(dim=1)
    M = 1
    real_MSE = ((x_src - x_real_rec) ** 2).mean(dim=(1, 2, 3))
    PSNR = 20 * torch.log10(torch.sqrt((M**2) / (real_MSE + 1e-8)))
    
    criterion = torchmetrics.StructuralSimilarityIndexMeasure(reduction='none')
    ssim = criterion(x_src, x_real_rec)

    if use_z:
        print('usez_SNR: ', SNR.cpu())
        print('usez_PSNR: ', PSNR.mean().cpu())
        print('usez_SSIM: ', ssim.mean().cpu())
    else:
        print('enc_SNR: ', SNR.cpu())
        print('enc_PSNR: ', PSNR.mean().cpu())
        print('enc_SSIM: ', ssim.mean().cpu())
    
    return PSNR

def make_anomaly_heatmap(anomaly_branch, x_src, flip_rgb=True, show=False):
    device = x_src.device
    if x_src.shape[1] == 1:
        anomaly_branch = torch.cat((anomaly_branch,anomaly_branch,anomaly_branch),dim=1)
        x_src = torch.cat((x_src, x_src, x_src), dim=1)
    anomaly_branch = (anomaly_branch.abs())
    anomaly_branch = anomaly_branch.sum(1, keepdim=True).cpu().numpy().transpose(0, 2, 3, 1)
    anomaly_branch = np.clip(np.round(255 * anomaly_branch), 0, 255).astype(np.uint8)
    x_src = denormalize(x_src).cpu().numpy().transpose(0, 2, 3, 1)
    x_src = np.round(255 * x_src).astype(np.uint8)
    #img2 = cv2.merge((x_src,x_src,x_src))
    super_imposed_img = np.zeros_like(x_src)
    for ii in range(x_src.shape[0]):
        heatmap_img = cv2.applyColorMap(anomaly_branch[ii], cv2.COLORMAP_JET)
        if flip_rgb:
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
        if x_src[ii].shape[2] == 1:
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)
            heatmap_img = heatmap_img[..., np.newaxis]

        #heatmap_img[heatmap_img < 1/255] = x_src[ii][heatmap_img < 1/255]
        weight_img = cv2.addWeighted(heatmap_img, 0.5, x_src[ii], 0.5, 0)
        if show:
            cv2.namedWindow('heat map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('heat map', 512, 512)
            cv2.imshow('heat map', anomaly_branch[ii])
            cv2.waitKey(0)
            cv2.imshow('heat map', heatmap_img)
            cv2.waitKey(0)
            cv2.imshow('heat map', weight_img)
            cv2.waitKey(0)
        if x_src[ii].shape[2] == 1:
            weight_img = weight_img[..., np.newaxis]
        super_imposed_img[ii] = weight_img
    
    super_imposed_img = torch.from_numpy(((super_imposed_img.astype(np.float)/255)-0.5) / 0.5).permute(0, 3, 1, 2).to(device).float()
    return super_imposed_img

    

def tensor_contrast_stretch(x):
    N, C, H, W = x.size()
    M, _ = torch.max(x.reshape((N, C, H * W)), dim=2)
    m, _ = torch.min(x.reshape((N, C, H * W)), dim=2)
    M = M.unsqueeze(-1).unsqueeze(-1)
    m = m.unsqueeze(-1).unsqueeze(-1)
    y = (x - m)/(M - m + 1e-8)
    y = 2 * y - 1
    return y


@torch.no_grad()
def debug_image(nets, args, inputs, step, laptop_mode=False):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    filename = ospj(args.sample_dir, '%06d_test_psi' % (step))
    test_psi(nets, args, x_src, y_src, x_ref, y_ref, filename + '_usez', use_z=True, step=step)
    psnr = test_anomaly(nets, args, x_src, y_src, x_ref, filename + '_usez', use_z=True, batch_idx=0)
    

def tensor2ndarray255(images, denormalize=False):
    if denormalize:
        images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    else:
        images = torch.clamp(images, 0, 1)
    if len(images.shape)>3:
        return images.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255
    else:
        return images.cpu().detach().numpy().transpose(1, 2, 0) * 255
    

def contrast_stretch(x):
    M = np.max(x)
    m = np.min(x)
    x = (x - m) / (M - m + 1e-16)
    return x


def imshow_tensor(T, is_heatmap=False, display=True, denormalize=True):
    if display:
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    if denormalize:
        T = ((T+1)/2).clamp(0, 1)

    x = T.permute((1, 2, 0)).flip(2).detach().cpu().numpy()
    if is_heatmap:
        x = contrast_stretch(x)
        x = (np.round(x * 255)).astype(np.uint8)
        if x.shape[2] > 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.equalizeHist(x)
        x = (np.round((x/255)**2 * 255)).astype(np.uint8)
        h = cv2.applyColorMap(x, cv2.COLORMAP_OCEAN)

        if display:
            cv2.imshow('image', h)
            cv2.waitKey(0)
        return h
    else:
        x = contrast_stretch(x)
        if display:
            cv2.imshow('image', x)
            cv2.waitKey(0)
        return x
