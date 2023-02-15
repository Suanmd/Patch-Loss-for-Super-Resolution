import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
import copy

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    if opt['val']['draw_curves']:
        f_opt = copy.deepcopy(opt)
        epoch_max = osp.splitext(opt['path']['pretrain_network_g'])[-2].split('_')[-1]
        for epoch_i in range(opt['val']['iters_stride'], int(epoch_max)+1, opt['val']['iters_stride']):
            opt['path']['pretrain_network_g'] = f_opt['path']['pretrain_network_g'].replace(epoch_max, str(epoch_i))
            # create model
            model = build_model(opt) # also load pth.
            for test_loader in test_loaders:
                test_set_name = test_loader.dataset.opt['name']
                logger.info(f'Testing {test_set_name}...')
                model.validation(test_loader, current_iter=opt['name'], tb_logger=None, 
                                 draw_curves=opt['val']['draw_curves'], save_img=opt['val']['save_img'])
    else:
        # create model
        model = build_model(opt) # also load pth.
        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            logger.info(f'Testing {test_set_name}...')
            model.validation(test_loader, current_iter=opt['name'], tb_logger=None, 
                             draw_curves=opt['val']['draw_curves'], save_img=opt['val']['save_img'])

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
