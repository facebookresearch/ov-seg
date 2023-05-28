# torchrun --nproc_per_node 8 -m training.main \
#     --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_1cap_nouns_mIoU20.csv \
#     --train-num-samples 442117 \
#     --lr 0.000005 \
#     --warmup 100 \
#     --force-quick-gelu \
#     --dataset-type csv \
#     --batch-size 128 \
#     --precision amp \
#     --workers 4 \
#     --model  ViT-B-16 \
#     --lock-text \
#     --zeroshot-frequency 1 \
#     --save-frequency 5 \
#     --epoch 5 \
#     --pretrained  openai \
#     --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask


torchrun --nproc_per_node 8 -m training.main \
    --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_1cap_nouns_mIoU20.csv \
    --train-num-samples 442117 \
    --lr 0.000005 \
    --warmup 100 \
    --force-quick-gelu \
    --dataset-type csv \
    --batch-size 128 \
    --precision amp \
    --workers 4 \
    --model  ViT-B-16 \
    --lock-text \
    --zeroshot-frequency 1 \
    --save-frequency 5 \
    --epoch 32 \
    --pretrained  openai \
    --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask


#torchrun --nproc_per_node 8 -m training.main \
#    --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_1cap_nouns_mIoU20.csv \
#    --train-num-samples 442117 \
#    --lr 0.000005 \
#    --warmup 100 \
#    --force-quick-gelu \
#    --dataset-type csv \
#    --batch-size 128 \
#    --precision amp \
#    --workers 4 \
#    --model  ViT-B-16 \
#    --ema \
#    --ema_ratio 0.999 \
#    --lock-text \
#    --zeroshot-frequency 1 \
#    --save-frequency 5 \
#    --epoch 5 \
#    --pretrained  openai \
#    --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask
#
#torchrun --nproc_per_node 8 -m training.main \
#    --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_1cap_nouns_mIoU20.csv \
#    --train-num-samples 442117 \
#    --lr 0.000005 \
#    --warmup 100 \
#    --force-quick-gelu \
#    --dataset-type csv \
#    --batch-size 128 \
#    --precision amp \
#    --workers 4 \
#    --model  ViT-B-16 \
#    --ema \
#    --ema_ratio 0.99 \
#    --lock-text \
#    --zeroshot-frequency 1 \
#    --save-frequency 5 \
#    --epoch 5 \
#    --pretrained  openai \
#    --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask


# torchrun --nproc_per_node 8 -m training.main \
#     --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_merge_nouns_mIoU27_deduplicated.csv \
#     --train-num-samples 1341804 \
#     --lr 0.000005 \
#     --warmup 100 \
#     --force-quick-gelu \
#     --dataset-type csv \
#     --batch-size 128 \
#     --precision amp \
#     --workers 4 \
#     --model  ViT-B-16 \
#     --zeroshot-frequency 1 \
#     --save-frequency 5 \
#     --epoch 5 \
#     --pretrained  openai \
#     --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask

# torchrun --nproc_per_node 8 -m training.main \
#     --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_1cap_nouns_mIoU20_appear41+.csv \
#     --train-num-samples 394218 \
#     --lr 0.000005 \
#     --warmup 100 \
#     --force-quick-gelu \
#     --dataset-type csv \
#     --batch-size 128 \
#     --precision amp \
#     --workers 4 \
#     --model  ViT-B-16 \
#     --lock-text \
#     --zeroshot-frequency 1 \
#     --save-frequency 5 \
#     --epoch 5 \
#     --pretrained  openai \
#     --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask
#
#
# torchrun --nproc_per_node 8 -m training.main \
#     --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_1cap_nouns_mIoU20_max1000.csv \
#     --train-num-samples 319493 \
#     --lr 0.000005 \
#     --warmup 100 \
#     --force-quick-gelu \
#     --dataset-type csv \
#     --batch-size 128 \
#     --precision amp \
#     --workers 4 \
#     --model  ViT-B-16 \
#     --lock-text \
#     --zeroshot-frequency 1 \
#     --save-frequency 5 \
#     --epoch 5 \
#     --pretrained  openai \
#     --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask
#
# torchrun --nproc_per_node 8 -m training.main \
#     --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_1cap_nouns_mIoU20_max100.csv \
#     --train-num-samples 132425 \
#     --lr 0.000005 \
#     --warmup 100 \
#     --force-quick-gelu \
#     --dataset-type csv \
#     --batch-size 128 \
#     --precision amp \
#     --workers 4 \
#     --model  ViT-B-16 \
#     --lock-text \
#     --zeroshot-frequency 1 \
#     --save-frequency 5 \
#     --epoch 5 \
#     --pretrained  openai \
#     --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask

# torchrun --nproc_per_node 8 -m training.main \
#     --train-data /home/jeffliang/zsseg/datasets/coco/meta/coco_merge_nouns_mIoU27_deduplicated.csv \
#     --train-num-samples 1341804 \
#     --lr 0.000005 \
#     --warmup 100 \
#     --force-quick-gelu \
#     --dataset-type csv \
#     --batch-size 128 \
#     --precision amp \
#     --workers 4 \
#     --model  ViT-B-16 \
#     --lock-text \
#     --lock-image \
#     --lock-image-unlocked-groups 1 \
#     --zeroshot-frequency 1 \
#     --save-frequency 5 \
#     --epoch 5 \
#     --pretrained  openai \
#     --imagenet-val /home/jeffliang/zsseg/datasets/ADEChallengeData2016/images/validation_mask
#