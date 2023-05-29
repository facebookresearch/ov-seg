import torch
from collections import OrderedDict

# PATH to trained MaskFormer model (containing a CLIP classifier)
ovseg_model = torch.load('/home/jeffliang/ov-seg/weights/ovseg_swinbase_vitL14_mpt_only.pth', 'cpu')

# PATH to finetuned CLIP weights
clip = OrderedDict()
new_clip_model = torch.load('/home/jeffliang/ov-seg/open_clip_training/src/logs/2023_05_28-22_15_08-model_ViT-L-14-lr_5e-06-b_32-j_4-p_amp/checkpoints/epoch_1.pt', 'cpu')

new_clip_model = new_clip_model['state_dict']
new_clip = OrderedDict()


for k, v in new_clip_model.items():
    k = k.replace('module.', '')
    # if not k == 'visual.mask_embedding':
    new_clip[k] = v
    new_k = 'clip_adapter.clip_model.' + k
    if new_k in ovseg_model['model'].keys():
        clip[k] = ovseg_model['model'][new_k]
    else:
        print(f'{new_k} does not exist in ckpt')

for c_p, new_c_p in zip(clip.items(), new_clip.items()):
    k1, v1 = c_p
    k2, v2 = new_c_p
    assert k1 == k2
    diff = (v1-v2).abs().mean()
    print(f'{k1} difference {diff}')