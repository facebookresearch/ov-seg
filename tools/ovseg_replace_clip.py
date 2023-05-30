# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from collections import OrderedDict


# PATH to finetune clip model
clip_ckpt = torch.load('/home/jeffliang/ov-seg/open_clip_training/src/logs/2023_05_29-16_27_38-mask_prompt_tuning-model_ViT-L-14-lr_0.05-b_32-j_4-p_amp/checkpoints/epoch_2.pt')

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
try:
    ovseg_model['model']['clip_adapter.clip_model.visual.mask_embedding'] = new_model['visual.mask_embedding']
    print('clip_ckpt has mask_embedding, remember to set MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD True during OVSeg evaluation')
except:
    print('clip_ckpt does not have mask_embedding, remember to set MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD False during OVSeg evaluation')

torch.save(ovseg_model, '/home/jeffliang/ov-seg/weights/new_ovseg.pth')
