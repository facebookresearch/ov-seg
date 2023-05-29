# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from collections import OrderedDict


# PATH to finetune clip model
clip_ckpt = torch.load('/home/jeffliang/ov-seg/open_clip_training/src/logs/2023_05_28-23_35_23-model_ViT-L-14-lr_5e-06-b_32-j_4-p_amp/checkpoints/epoch_5.pt')

new_model = OrderedDict()
state_dict = clip_ckpt['state_dict']

for k, v in state_dict.items():
    new_key = k.replace('module.','')
    new_model[new_key] = v

# PATH to trained MaskFormer model
ovseg_model = torch.load('/home/jeffliang/ov-seg/weights/ovseg_swinbase_vitL14_mpt_only.pth', 'cpu')

for k, v in new_model.items():
    new_k = 'clip_adapter.clip_model.' + k
    if new_k in ovseg_model['model'].keys():
        ovseg_model['model'][new_k] = v
    else:
        print(f'{new_k} does not exist in ckpt')

# ovseg_model['model']['clip_adapter.clip_model.visual.mask_embedding'] = new_model['visual.mask_embedding']

torch.save(ovseg_model, '/home/jeffliang/ov-seg/weights/ovseg_swinbase_vitL14_ft.pth')
