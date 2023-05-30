import logging
from contextlib import suppress
import inspect

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm

from open_clip import tokenize
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from .ade150_zeroshot_data import ade150_classnames


def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenize(texts).to(args.device)  # tokenize
            if args.distributed and not args.horovod:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk], pred[0]


def run(model, classifier, dataloader, args):
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        preds = []
        targets = []
        macc = Accuracy('multiclass', num_classes=150, average='macro').cuda()
        for batch, target in tqdm(dataloader, unit_scale=args.batch_size):
            if args.with_mask:
                images, masks = batch
                masks = masks.to(args.device)
            else:
                images = batch
            images = images.to(args.device)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    if args.with_mask:
                        image_features = model.module.encode_image(images, masks)
                    else:
                        image_features = model.module.encode_image(images)
                else:
                    if args.with_mask:
                        image_features = model.encode_image(images, masks)
                    else:
                        image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            (acc1, acc5), pred = accuracy(logits, target, topk=(1, 5))
            preds.append(pred)
            targets.append(target)
            top1 += acc1
            top5 += acc5
            n += images.size(0)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5, macc(preds, targets).item()


def zero_shot_eval(model, data, epoch, args):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'ade-val' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')
    # for i in range(len(openai_imagenet_template)):
    #     template = openai_imagenet_template[i]
    #     logging.info(inspect.getsource(template))
    logging.info('Building zero-shot classifier')
    if 'ade-val' in data:
        classifier = zero_shot_classifier(model, ade150_classnames, openai_imagenet_template, args)
    else:
        classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5, macc = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
        results['mean-accuracy-top1'] = macc
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    if 'ade-val' in data:
        top1, top5, macc = run(model, classifier, data['ade-val'].dataloader, args)
        results['ade150-zeroshot-val-top1'] = top1
        results['ade150-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
