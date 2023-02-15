import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('patch_opt'):
            self.cri_patch = build_loss(train_opt['patch_opt']).to(self.device)
        else:
            self.cri_patch = None

        if train_opt.get('patch_opt_3d'):
            self.cri_patch_3d = build_loss(train_opt['patch_opt_3d']).to(self.device)
        else:
            self.cri_patch_3d = None

        if train_opt.get('patch_opt_3d_xd'):
            self.cri_patch_3d_xd = build_loss(train_opt['patch_opt_3d_xd']).to(self.device)
        else:
            self.cri_patch_3d_xd = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_patch is None and self.cri_patch_3d is None and self.cri_patch_3d_xd is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # patch loss
        if self.cri_patch:
            l_patch = self.cri_patch(self.output, self.gt)
            l_total += l_patch
            loss_dict['l_patch'] = l_patch

        if self.cri_patch_3d:
            l_patch_3d = self.cri_patch_3d(self.output, self.gt)
            l_total += l_patch_3d
            loss_dict['l_patch_3d'] = l_patch_3d

        if self.cri_patch_3d_xd:
            l_patch_3d_xd = self.cri_patch_3d_xd(self.output, self.gt)
            l_total += l_patch_3d_xd
            loss_dict['l_patch_3d_xd'] = l_patch_3d_xd

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, draw_curves, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, draw_curves, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, draw_curves, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                _our_metrics = {}
                for name, opt_ in self.opt['val']['metrics'].items():
                    _temp_value = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += _temp_value
                    _our_metrics[name] = _temp_value

                if 'psnr' in _our_metrics.keys():
                    _psnr = round(_our_metrics["psnr"], 3)
                if 'ssim' in _our_metrics.keys():
                    _ssim = round(_our_metrics["ssim"], 4)
                if 'lpips' in _our_metrics.keys():
                    _lpips = round(_our_metrics["lpips"], 4)
                if 'niqe' in _our_metrics.keys():
                    _niqe = round(_our_metrics["niqe"], 3)
                if 'patch' in _our_metrics.keys():
                    _patch = round(_our_metrics["patch"], 3)
                if 'patch2' in _our_metrics.keys():
                    _patch2 = round(_our_metrics["patch2"], 2)

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        # save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                        #                          f'{img_name}_{self.opt["name"]}.png')
                        _all_metrics_to_str = str(_psnr) + '_' + str(_ssim) + '_' + str(_lpips) + \
                                          '_'+str(_niqe) + '_' + str(_patch) + '_' + str(_patch2)
                        save_img_path = osp.join(self.opt['path']['visualization'], 
                                                 dataset_name, 
                                                 f'{img_name}_'+_all_metrics_to_str+'.png')                
                imwrite(sr_img, save_img_path)
  
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        # change the folder name
        if save_img:
            import os
            _new_name = '_' + str(round(self.metric_results['psnr'], 3)) + \
                        '_' + str(round(self.metric_results['ssim'], 4)) + \
                        '_' + str(round(self.metric_results['lpips'], 4)) + \
                        '_' + str(round(self.metric_results['niqe'], 3)) + \
                        '_' + str(round(self.metric_results['patch'], 3)) + \
                        '_' + str(round(self.metric_results['patch2'], 2)) + \
                        '_' + osp.splitext(self.opt['path']['pretrain_network_g'])[-2].split('_')[-1]
            os.rename(osp.join(self.opt['path']['visualization'], dataset_name),
                      osp.join(self.opt['path']['visualization'], dataset_name+_new_name))

        if not save_img and 'draw_curves' in self.opt['val'].keys():
            # create a file and record the metrics
            _txt_path = self.opt['path']['visualization'].replace('visualization','') + 'metrics.txt'

            _txt_content = '\n' + 'data: ' + dataset_name + \
                           '\t' + 'iters: ' + osp.splitext(self.opt['path']['pretrain_network_g'])[-2].split('_')[-1] + \
                           '\t' + 'psnr: ' + str(round(self.metric_results['psnr'], 3)) + \
                           '\t' + 'ssim: ' + str(round(self.metric_results['ssim'], 4)) + \
                           '\t' + 'lpips: ' + str(round(self.metric_results['lpips'], 4)) + \
                           '\t' + 'niqe: ' + str(round(self.metric_results['niqe'], 3)) + \
                           '\t' + 'patch: ' + str(round(self.metric_results['patch'], 3)) + \
                           '\t' + 'patch2: ' + str(round(self.metric_results['patch2'], 2))
            with open(_txt_path,"a") as ff:
                ff.write(_txt_content)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
