import cv2
import numpy as np
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY
import torch

@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
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

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
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

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()



@METRIC_REGISTRY.register()
def calculate_pearson_patch(img, img2, crop_border, kernels=[3, 5, 7], input_order='HWC', test_y_channel=True, **kwargs):
    """Calculate PATCH.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
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

    IMG = torch.tensor(img/255.).squeeze().unsqueeze(0).unsqueeze(0).to('cuda')
    IMG2 = torch.tensor(img2/255.).squeeze().unsqueeze(0).unsqueeze(0).to('cuda')
    from basicsr.losses.losses import PatchesKernel

    pearsons = []
    for _kernel in kernels:
        _patchkernel = PatchesKernel(_kernel, 1).to('cuda')
        img = _patchkernel(IMG)                  # [N, patch_num, patch_len ** 2]
        img2 = _patchkernel(IMG2)                # [N, patch_num, patch_len ** 2]
        img = img.reshape(-1, img.shape[-1])     # [N * patch_num, patch_len ** 2]
        img2 = img2.reshape(-1, img2.shape[-1])  # [N * patch_num, patch_len ** 2]
        x = torch.clamp(img - torch.mean(img, dim=1, keepdim=True), 0.000001, 0.999999)   # [N * patch_num, patch_len ** 2]
        y = torch.clamp(img2 - torch.mean(img2, dim=1, keepdim=True), 0.000001, 0.999999) # [N * patch_num, patch_len ** 2]
        dot_x_y = torch.einsum('ik,ik->i',x,y)   # [N * patch_num]
        pearson_x_y = torch.mean(torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1))))
        # First the mean, then the exp
        pearsons.append(pearson_x_y)
    Pearsons = torch.stack(pearsons)
    return float(torch.exp(torch.mean(Pearsons)).detach().cpu().numpy())

def get_cot_similar_matrix(v1, v2):
    num = torch.einsum('ik,ik->i', v1, v2)
    denom = torch.mul(torch.norm(v1, dim=1), torch.norm(v2, dim=1))
    res = torch.clamp(torch.div(num, denom), 0.000001, 0.999999)
    return torch.mean(torch.div(res, (torch.sqrt(1 - res**2))))
    # return torch.mean(res)

@METRIC_REGISTRY.register()
def calculate_cosine_patch(img, img2, crop_border, kernels=[3, 5, 7], input_order='HWC', test_y_channel=True, **kwargs):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
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

    IMG = torch.tensor(img/255.).squeeze().unsqueeze(0).unsqueeze(0).to('cuda')
    IMG2 = torch.tensor(img2/255.).squeeze().unsqueeze(0).unsqueeze(0).to('cuda')
    from basicsr.losses.losses import PatchesKernel

    cos_thetas = []
    for _kernel in kernels:
        # _patchkernel = PatchesKernel(_kernel, _kernel//2).to('cuda')
        _patchkernel = PatchesKernel(_kernel, 1).to('cuda')
        img = _patchkernel(IMG)                  # [N, patch_num, patch_len ** 2]
        img2 = _patchkernel(IMG2)                # [N, patch_num, patch_len ** 2]
        img = img.reshape(-1, img.shape[-1])     # [N * patch_num, patch_len ** 2]
        img2 = img2.reshape(-1, img2.shape[-1])  # [N * patch_num, patch_len ** 2]
        # x = img - torch.mean(img, dim=1, keepdim=True) + 0.000001  # [N * patch_num, patch_len ** 2]
        # y = img2 - torch.mean(img2, dim=1, keepdim=True) + 0.000001 # [N * patch_num, patch_len ** 2]
        x = img  # [N * patch_num, patch_len ** 2]
        y = img2 # [N * patch_num, patch_len ** 2]
        cos_thetas.append(get_cot_similar_matrix(x, y))

    Cos_thetas = torch.stack(cos_thetas)
    return float(torch.mean(Cos_thetas).detach().cpu().numpy())


@METRIC_REGISTRY.register()
def calculate_cosine_xd_patch(img, img2, crop_border, kernels=[3, 5, 7], input_order='HWC', test_y_channel=True, **kwargs):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
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

    IMG = torch.tensor(img/255.).squeeze().unsqueeze(0).unsqueeze(0).to('cuda')
    IMG2 = torch.tensor(img2/255.).squeeze().unsqueeze(0).unsqueeze(0).to('cuda')
    from basicsr.losses.losses import PatchesKernel

    cos_thetas = []
    for _kernel in kernels:
        # _patchkernel = PatchesKernel(_kernel, _kernel//2).to('cuda')
        _patchkernel = PatchesKernel(_kernel, 1).to('cuda')
        img = _patchkernel(IMG)                  # [N, patch_num, patch_len ** 2]
        img2 = _patchkernel(IMG2)                # [N, patch_num, patch_len ** 2]
        img = img.reshape(-1, img.shape[-1])     # [N * patch_num, patch_len ** 2]
        img2 = img2.reshape(-1, img2.shape[-1])  # [N * patch_num, patch_len ** 2]
        # x = img - torch.mean(img, dim=1, keepdim=True) + 0.000001  # [N * patch_num, patch_len ** 2]
        # y = img2 - torch.mean(img2, dim=1, keepdim=True) + 0.000001 # [N * patch_num, patch_len ** 2]
        x = img                                  # [N * patch_num, patch_len ** 2]
        y = img2                                 # [N * patch_num, patch_len ** 2]
        cos_thetas.append(get_cot_similar_matrix_xd(x, y))

    Cos_thetas = torch.stack(cos_thetas)
    return float(torch.mean(Cos_thetas).detach().cpu().numpy())

def get_cot_similar_matrix_xd(v1, v2):
    num = torch.einsum('ik,ik->i', v1, v2)
    denom = torch.mul(torch.norm(v1, dim=1), torch.norm(v2, dim=1))
    res = torch.clamp(torch.div(num, denom), 0.000001, 0.999999)
    dy = torch.clamp(torch.std(v2, dim=1), 0., 1.)
    return torch.mean(torch.mul(dy, torch.div(res, (torch.sqrt(1 - res**2)))))