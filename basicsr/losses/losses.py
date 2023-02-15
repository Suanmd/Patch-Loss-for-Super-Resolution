import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
import numpy as np
from basicsr.metrics.metric_util import reorder_image, to_y_channel

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 perceptual_patch_weight=1.0,
                 style_weight=0.,
                 criterion='patch',
                 perceptual_kernels=[4,8],
                 use_std_to_force=True):

        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.patch_weights = perceptual_patch_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.perceptual_kernels = perceptual_kernels
        self.use_std_to_force = use_std_to_force
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        elif self.criterion_type == 'patch':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                elif self.criterion_type == 'patch':
                    if self.patch_weights == 0:
                        percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
                    else:
                        percep_loss += self.patch(x_features[k], gt_features[k], self.use_std_to_force) * self.layer_weights[k] * self.patch_weights + self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def patch(self, x, gt, use_std_to_force):
        loss = 0.
        for _kernel in self.perceptual_kernels:
            _patchkernel3d = PatchesKernel3D(_kernel, _kernel//2).to('cuda')   # create instance
            x_trans = _patchkernel3d(x)
            gt_trans = _patchkernel3d(gt)
            x_trans = x_trans.reshape(-1, x_trans.shape[-1])
            gt_trans = gt_trans.reshape(-1, gt_trans.shape[-1])
            dot_x_y = torch.einsum('ik,ik->i', x_trans, gt_trans)
            if use_std_to_force == False:
                cosine0_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x_trans ** 2, dim=1))), torch.sqrt(torch.sum(gt_trans ** 2, dim=1)))
                loss = loss + torch.mean(1-cosine0_x_y) # y = 1-x
            else:
                dy = torch.std(gt_trans, dim=1)
                cosine_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x_trans ** 2, dim=1))), torch.sqrt(torch.sum(gt_trans ** 2, dim=1)))
                cosine_x_y_d = torch.mul((1-cosine_x_y), dy) # y = (1-x)dy
                loss = loss + torch.mean(cosine_x_y_d)
        return loss

class PatchesKernel3D(nn.Module):
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
        x = x.reshape(batchsize*channels, x.shape[-2], x.shape[-1]).unsqueeze(1)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, channels, -1, self.kernelsize ** 2).permute(0, 2, 1, 3)
        return x

@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operation: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_d = len(pred_fake)
        loss = 0
        for i in range(num_d):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_d
        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class patchLoss(nn.Module):
    """Define patch loss

    Args:
        kernel_sizes (list): add (x, y) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    def __init__(self, kernel_sizes=[2, 4], loss_weight=1.0):
        super(patchLoss, self).__init__()
        self.kernels = kernel_sizes
        self.loss_weight = loss_weight

    def forward(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        # from basicsr.utils.matlab_functions import rgb2ycbcr

        # PIL proves that the data format is RGB
        # from PIL import Image
        # for i_ in range(16):
        #     label = labels[i_,:,:,:] # torch [3, 96, 96]
        #     label9 = (label * 255).detach().cpu().numpy().astype(np.uint8).transpose(1,2,0) # [96, 96, 3]
        #     im = Image.fromarray(label9)
        #     im.save(str(i_)+'.png')

        # for j_ in range(16):
        #     label = labels[j_,:,:,:]
        #     label9 = (label * 255).detach().cpu().numpy().astype(np.uint8).transpose(1,2,0) # [96, 96, 3]
        #     _a  =  np.expand_dims(label9[:,:,2],axis=-1)
        #     _b  =  np.expand_dims(label9[:,:,1],axis=-1)
        #     _c  =  np.expand_dims(label9[:,:,0],axis=-1)
        #     labelx = np.concatenate([_a,_b,_c], axis = -1)
        #     im = Image.fromarray(labelx)
        #     im.save(str(j_)+'xx.png')

        # labels -> [16,3,96,96]
        # label0 = labels[1,:,:,:] # label0 -> [3,96,96]
        # label3 = label0.unsqueeze(-1).permute(3,1,2,0).squeeze() # label3 -> [96,96,3]
        # label4 = label0.unsqueeze(-1).transpose(0,3).squeeze() # label4 -> [96,96,3]
        # label6 = (16. + (65.481 * label3[:, :, 0] + 128.553 * label3[:, :, 1] + 24.966 * label3[:, :, 2]))/255.  # [96, 96]

        # label1 = label0.detach().cpu().numpy().astype(np.float32) # [3, 96, 96]
        # label2 = reorder_image(label1, input_order='CHW')  # [96, 96, 3]
        # label5 = rgb2ycbcr(label2, y_only=True)  # [96, 96]
        # # Label5 and Label6 are equal in value.

        # labels7 = (16. + (65.481 * labels[:, 0, :, :] + 128.553 * labels[:, 1, :, :] + 24.966 * labels[:, 2, :, :]))/255. # [16, 96, 96]
        # # Only one line can do this: 
        # # labels7[1] == label6 == label5

        preds = (16. + (65.481 * preds[:, 0, :, :] + 128.553 * preds[:, 1, :, :] + 24.966 * preds[:, 2, :, :]))/255.
        labels = (16. + (65.481 * labels[:, 0, :, :] + 128.553 * labels[:, 1, :, :] + 24.966 * labels[:, 2, :, :]))/255.
        preds = preds.unsqueeze(1)
        labels = labels.unsqueeze(1) 
        loss = 0.

        for _kernel in self.kernels:
            _patchkernel = PatchesKernel(_kernel, _kernel//2 + 1).to('cuda')           # create instance
            preds_trans = _patchkernel(preds)                                          # [N, patch_num, patch_len ** 2]
            labels_trans = _patchkernel(labels)                                        # [N, patch_num, patch_len ** 2]
            preds_trans = preds_trans.reshape(-1, preds_trans.shape[-1])               # [N * patch_num, patch_len ** 2]
            labels_trans = labels_trans.reshape(-1, labels_trans.shape[-1])            # [N * patch_num, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum('ik,ik->i',x,y)                                     # [N * patch_num]
            pearson_x_y = torch.mean(torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1))))
            loss = loss + torch.exp(-pearson_x_y) # y = e^(-x)
            # loss = loss - pearson_x_y # y = - x

        return loss * self.loss_weight

class PatchesKernel(nn.Module):
    def __init__(self, kernelsize, kernelstride, kernelpadding=0):
        super(PatchesKernel, self).__init__()
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
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, -1, self.kernelsize ** 2)
        return x


@LOSS_REGISTRY.register()
class patchLoss3D(nn.Module):
    """Define patch loss

    Args:
        kernel_sizes (list): add (4,2), (8,4), (16,8), (32,16) or (64,32) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    def __init__(self, kernel_sizes=[2, 4], loss_weight=1.0):
        super(patchLoss3D, self).__init__()
        self.kernels = kernel_sizes
        self.loss_weight = loss_weight

    def forward(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        loss = 0.
        for _kernel in self.kernels:
            _patchkernel = PatchesKernel3D(_kernel, _kernel//2 + 1).to('cuda') # create instance
            preds_trans = _patchkernel(preds)                                  # [N, patch_num, channels, patch_len ** 2]
            labels_trans = _patchkernel(labels)                                # [N, patch_num, channels, patch_len ** 2]
            preds_trans = preds_trans.reshape(-1, preds_trans.shape[-1])       # [N * patch_num * channels, patch_len ** 2]
            labels_trans = labels_trans.reshape(-1, labels_trans.shape[-1])    # [N * patch_num * channels, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum('ik,ik->i',x,y)                             # [N * patch_num]
            cosine0_x_y = torch.mean(torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1))))
            loss = loss + (1 - cosine0_x_y) # y = 1 - x
        return loss * self.loss_weight

@LOSS_REGISTRY.register()
class patchLoss3DXD(nn.Module):
    """Define patch loss

    Args:
        kernel_sizes (list): add (x, y) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    def __init__(self, kernel_sizes=[2, 4], loss_weight=1.0, use_std_to_force=True):
        super(patchLoss3DXD, self).__init__()
        self.kernels = kernel_sizes
        self.loss_weight = loss_weight
        self.use_std_to_force = use_std_to_force

    def forward(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        loss = 0.
        for _kernel in self.kernels:
            _patchkernel = PatchesKernel3D(_kernel, _kernel//2 + 1).to('cuda') # create instance
            preds_trans = _patchkernel(preds)                                  # [N, patch_num, channels, patch_len ** 2]
            labels_trans = _patchkernel(labels)                                # [N, patch_num, channels, patch_len ** 2]
            preds_trans = preds_trans.reshape(-1, preds_trans.shape[-1])       # [N * patch_num * channels, patch_len ** 2]
            labels_trans = labels_trans.reshape(-1, labels_trans.shape[-1])    # [N * patch_num * channels, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum('ik,ik->i',x,y)      
            if self.use_std_to_force == False:
                cosine0_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1)))
                loss = loss + torch.mean((1-cosine0_x_y)) # y = 1-x
            else:
                dy = torch.std(labels_trans*10, dim=1)
                cosine_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1)))
                cosine_x_y_d = torch.mul((1-cosine_x_y), dy) # y = (1-x) dy
                loss = loss + torch.mean(cosine_x_y_d) 
        return loss * self.loss_weight
