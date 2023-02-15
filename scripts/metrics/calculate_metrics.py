import argparse
import cv2
import numpy as np
from os import path as osp
import os

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir, imwrite, get_root_logger, img2tensor
from basicsr.utils.matlab_functions import bgr2ycbcr
import lpips
import glob
import shutil
from torchvision.transforms.functional import normalize
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.metrics.niqe import *

def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate LPIPS.
    Ref: https://github.com/xinntao/BasicSR/pull/367
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: LPIPS result.
    """
    assert img.shape == img2.shape, (f'Image shapes are differnet: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    # start calculating LPIPS metrics
    # loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False)  # RGB, normalized to [-1,1]
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).cuda()  # RGB, normalized to [-1,1]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    img_gt = img2 / 255.
    img_restored = img / 255.
  
    img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
    # norm to [-1, 1]
    normalize(img_gt, mean, std, inplace=True)
    normalize(img_restored, mean, std, inplace=True)

    # calculate lpips
    img_gt = img_gt.cuda()
    img_restored = img_restored.cuda()
    loss_fn_vgg.eval()
    lpips_val = loss_fn_vgg(img_restored.unsqueeze(0), img_gt.unsqueeze(0))

    # return lpips_val.detach().numpy().mean() 
    return lpips_val.detach().cpu().numpy().mean() 


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all = []
    ssim_all = []
    lpips_all = []
    niqe_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(args.restored, recursive=True, full_path=True)))

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    _M = ''
    for m in args.restored.split('/')[:-1]:
        _M +=  m + '/'
    _M += 'visualization/'
    if not os.path.exists(_M):
        os.mkdir(_M)
    _M += args.restored.split('/')[-1]
    if not os.path.exists(_M):
        os.mkdir(_M)



    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if args.suffix == '':
            img_path_restored = img_list_restored[i]
        else:
            img_path_restored = osp.join(args.restored, basename + args.suffix + ext)
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        _psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        _ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        _lpips = calculate_lpips(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        _niqe = calculate_niqe(img_restored * 255, crop_border=args.crop_border)
        # print(f'{i+1:3d}: {basename:25}. \tPSNR: {_psnr:.6f} dB, \tSSIM: {_ssim:.6f}')
        psnr_all.append(_psnr)
        ssim_all.append(_ssim)
        lpips_all.append(_lpips)
        niqe_all.append(_niqe)

        oldname = args.restored + '/' + basename + '_SwinIR.png'
        save_name_per_image = basename + '_' + str(round(_psnr, 3)) + '_' + str(round(_ssim, 4)) + '_' + str(round(_lpips, 4)) + '_' + str(round(_niqe, 3)) + '_0.000_0.00.png'
        # newname = args.restored + '/' + save_name_per_image
        # os.rename(oldname, newname)

        newname = _M + '/' + save_name_per_image
        shutil.move(oldname, newname)
        # shutil.copy(oldname,newname)
    os.rmdir(args.restored)

    mean_psnr = round(sum(psnr_all) / len(psnr_all), 3)
    mean_ssim = round(sum(ssim_all) / len(ssim_all), 4)
    mean_lpips = round(sum(lpips_all) / len(lpips_all), 4)
    mean_niqe = round(sum(niqe_all) / len(niqe_all), 3)

    oldname_dir = _M
    newname_picdir = _M.split('/')[-1] + '_' + str(mean_psnr) + '_' + str(mean_ssim) + '_' + str(mean_lpips) + '_' + str(mean_niqe) + '_0.000_0.00_500000'
    newname_dir = oldname_dir.replace(_M.split('/')[-1], '') + newname_picdir
    os.rename(oldname_dir, newname_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='datasets/val_set14/Set14', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default='results/Set14', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)
