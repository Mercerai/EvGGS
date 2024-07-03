from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
from . import yacs
from datetime import datetime
from pathlib import Path

cfg = CN()
cfg.task = 'hello'
cfg.gpus = [0]
cfg.exp_name = 'depth_pred'

cfg.record = CN()

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    # if -1 not in cfg.gpus:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    if 'bbox' in cfg:
        bbox = np.array(cfg.bbox).reshape((2, 3))
        center, half_size = np.mean(bbox, axis=0), (bbox[1]-bbox[0]).max().item() / 2.
        bbox = np.stack([center-half_size, center+half_size])
        cfg.bbox = bbox.reshape(6).tolist()

    print('EXP NAME: ', cfg.exp_name)

    cfg.local_rank = args.local_rank

    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'

def make_cfg(args):
    def merge_cfg(cfg_file, cfg):
        with open(cfg_file, 'r') as f:
            current_cfg = yacs.load_cfg(f)
        if 'parent_cfg' in current_cfg.keys():
            cfg = merge_cfg(current_cfg.parent_cfg, cfg)
            cfg.merge_from_other_cfg(current_cfg)
        else:
            cfg.merge_from_other_cfg(current_cfg)
        print(cfg_file)
        return cfg
    cfg_ = merge_cfg(args.cfg_file, cfg)
    try:
        index = args.opts.index('other_opts')
        cfg_.merge_from_list(args.opts[:index])
    except:
        cfg_.merge_from_list(args.opts)
    parse_cfg(cfg_, args)
    return cfg_

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/Ev3D_pretrain.yaml", type=str)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

cfg = make_cfg(args)

dt = datetime.today()
cfg.exp_name = '%s_%s%s' % (cfg.exp_name, str(dt.month).zfill(2), str(dt.day).zfill(2))
cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
cfg.record.file_path = "experiments/%s/file" % cfg.exp_name

for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
    Path(path).mkdir(exist_ok=True, parents=True)