# **Patch Loss for Super-Resolution**
## Introduction

The challenge of single-image super-resolution (SISR) is to maintain the quality of the enlarged images. In recent years, most SISR methods tend to exploit massive data and complex structures to improve image fidelity with high training costs. However, images with high PSNR are not necessarily visually pleasing, and spending a high cost to improve PSNR may not be compatible with practical applications. To this end, we propose a generic multi-scale perceptual loss, called patch loss, which can effectively improve the visual quality of the generated images in a plug-and-play manner.

The experimental models include EDSR, RCAN, SRGAN, ESRGAN, and SwinIR. To evaluate the impact of the proposed loss on the generated images, we use a variety of perceptual metrics, i.e., LPIPS, NIQE, Ma, and PI, to assess the image quality. Extensive experiments show that the patch loss can further improve the perceptual quality of the generated images.


![The proposed patch loss](https://github.com/Suanmd/Patch-Loss-for-Super-Resolution/blob/main/utils/img/example.png)

## Instruction
Sincere thanks to the developers of the [BasicSR](https://github.com/XPixelGroup/BasicSR) project. After the configuration is complete, please add the modified files we provided in the correct locations. **We will upload more model configs to GitHub later**.

If you want to quickly add patch loss to your own model, you can refer to the following example:

    import torch
    from torch import nn
    import torch.nn.functional as F
    
    class PatchesKernel(nn.Module):
        def __init__(self, kernelsize, kernelstride, kernelpadding=0):
            super(PatchesKernel3D, self).__init__()
            kernel = torch.eye(kernelsize ** 2).\
                view(kernelsize ** 2, 1, kernelsize, kernelsize)
            kernel = torch.FloatTensor(kernel)
            self.weight = nn.Parameter(data=kernel,
                                       requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(kernelsize ** 2),
                                     requires_grad=False)
            self.kernelsize = kernelsize
            self.stride = kernelstride
            self.padding = kernelpadding
    
        def forward(self, x):
            batchsize = x.shape[0]
            channels = x.shape[1]
            x = x.reshape(batchsize * channels, x.shape[-2],
                          x.shape[-1]).unsqueeze(1)
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
            x = x.permute(0, 2, 3, 1).reshape(batchsize, channels, -1,
                                      self.kernelsize ** 2).permute(0, 2, 1, 3)
            return x
    
    
    pk = PatchesKernel(kernelsize, kernelstride)
    output = pk(img)  # img shape: [b,c,h,w]


## Training

    python basicsr/train.py -opt options/train/EDSR/train_EDSR_Mx2.yml
    python basicsr/train.py -opt options/train/RCAN/train_RCAN_x2.yml
    python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRGAN_x4.yml
    python basicsr/train.py -opt options/train/ESRGAN/train_ESRGAN_x4.yml
    python basicsr/train.py -opt options/train/SwinIR/train_SwinIR_SRx4_scratch.yml

## Testing

    python basicsr/test.py -opt options/test/EDSR/test_EDSR_Mx2.yml
    python basicsr/test.py -opt options/test/RCAN/test_RCAN_x2.yml
    python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRGAN_x4.yml
    python basicsr/test.py -opt options/test/ESRGAN/test_ESRGAN_x4.yml
    bash ./options/test/SwinIR/SwinIRx2.sh

## Matlab Code

The MATLAB code provides two main functions *evaluate_results_dirs_linux.m* and *evaluate_results_dirs_win.m* to test the perceptual metrics NIQE, Ma, and PI. These two functions are designed to be executed on different systems.

    evaluate_results_dirs_linux test_results GT_datasets shave_width true
    evaluate_results_dirs_win test_results GT_datasets shave_width true

## FID Score

    python -m pytorch_fid path/to/dataset1 path/to/dataset2 --device cuda:0

## Reference Links

 1. [BasicSR](https://github.com/XPixelGroup/BasicSR)
 2. [PIRM2018](https://github.com/roimehrez/PIRM2018)
 3. [FID-Pytorch](https://github.com/mseitzer/pytorch-fid)
 4. [Ma et al.](https://github.com/chaoma99/sr-metric)
