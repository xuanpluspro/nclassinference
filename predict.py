"""CUDA-only SMP segmentation inference; one image at a time."""
import argparse
import csv
import io
import json
import logging
import time
from pathlib import Path
import numpy as np
from PIL import Image
from common import atomic, digest, images, write_json, save_image, colorize, overlay, read_mask, confusion, metrics, tile_starts, apply_dataset_config, indexed_image


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model-config', type=Path, help='Explicit config for metadata-free SMP weights')
    p.add_argument('--dataset-config', type=Path, help='Class-compatible visualization config, e.g. configs/riceseg.json')
    p.add_argument('--split', type=Path, help='One full extension-free image ID per line')
    p.add_argument('--gt', type=Path, help='Original-size indexed GT directory, same relative stem + .png')
    p.add_argument('--mode', choices=['resize','full','tile'], default='resize')
    p.add_argument('--size', type=int, help='Resize input side; default checkpoint training size, usually 512')
    p.add_argument('--tile-size', type=int, default=512)
    p.add_argument('--overlap', type=int, default=128)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--alpha', type=float, default=.45)
    p.add_argument('--jpg-quality', type=int, default=90)
    p.add_argument('--save-color', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-images', type=int, default=0)
    p.add_argument('--trust-checkpoint', action='store_true')
    g=p.add_mutually_exclusive_group(); g.add_argument('--resume',action='store_true'); g.add_argument('--overwrite',action='store_true')
    a=p.parse_args(argv)
    if a.max_images<0 or not 0<=a.alpha<=1 or not 1<=a.jpg_quality<=100: p.error('Invalid max-images/alpha/JPG quality')
    if a.size is not None and (a.size<32 or a.size%32): p.error('--size must be a multiple of 32')
    if a.tile_size<32 or a.tile_size%32 or not 0<=a.overlap<a.tile_size: p.error('Invalid tile-size/overlap')
    if not a.device.startswith('cuda'): p.error('CUDA required; no automatic CPU fallback')
    return a


def predict_image(model, image, cfg, args, torch):
    import torch.nn.functional as F
    device = torch.device(args.device)
    def forward(rgb, resized=None):
        w,h=rgb.size
        if resized: rgb=rgb.resize((resized,resized),Image.Resampling.BILINEAR)
        arr=np.asarray(rgb).copy()
        x=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device=device,dtype=torch.float32)/255
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]; std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        x=(x-mean)/std; ih,iw=x.shape[-2:]
        ph=max(32,(ih+31)//32*32); pw=max(32,(iw+31)//32*32)
        x=F.pad(x,(0,pw-iw,0,ph-ih))
        with torch.autocast('cuda',enabled=args.amp): logits=model(x)
        if not torch.is_tensor(logits) or logits.ndim!=4 or logits.shape[1]!=cfg['num_classes']: raise ValueError('Unexpected model output')
        if logits.shape[-2:]!=(ph,pw): logits=F.interpolate(logits.float(),size=(ph,pw),mode='bilinear',align_corners=False)
        logits=logits[:,:,:ih,:iw].float()
        if (ih,iw)!=(h,w): logits=F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)
        return logits.softmax(1)[0].cpu().numpy()
    with torch.inference_mode():
        if args.mode != 'tile':
            size = args.size or int(cfg.get('size',512))
            if size<32 or size%32: raise ValueError('Checkpoint size must be a positive multiple of 32')
            return forward(image, size if args.mode=='resize' else None).argmax(0).astype(np.uint8)
        w,h=image.size
        accum=np.zeros((cfg['num_classes'],h,w),dtype=np.float32); count=np.zeros((h,w),dtype=np.float32)
        for top in tile_starts(h,args.tile_size,args.overlap):
            for left in tile_starts(w,args.tile_size,args.overlap):
                right=min(w,left+args.tile_size); bottom=min(h,top+args.tile_size)
                prob=forward(image.crop((left,top,right,bottom)))
                accum[:,top:bottom,left:right]+=prob; count[top:bottom,left:right]+=1
        return (accum/count[None]).argmax(0).astype(np.uint8)


def run(args):
    import torch
    from model_loader import load
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Use the GPU instance with its existing GPU PyTorch.')
    device=torch.device(args.device); torch.cuda.set_device(device)
    root, entries=images(args.input,args.split); out=args.output.resolve()
    if out==root or root in out.parents: raise ValueError('Output must be outside input directory')
    if args.gt and (out==args.gt.resolve() or args.gt.resolve() in out.parents): raise ValueError('Output must be outside GT directory')
    manifest_path=out/'meta/run.json'
    if out.exists() and any(out.iterdir()) and not(args.resume or args.overwrite): raise FileExistsError('Use a new output, --resume, or --overwrite')
    model,cfg=load(args.weights,args.trust_checkpoint,args.model_config)
    if args.dataset_config: cfg=apply_dataset_config(cfg,args.dataset_config)
    model=model.to(device).eval()
    signature=dict(weights_sha256=digest(args.weights),input=str(root),config=cfg,mode=args.mode,size=args.size,
                   tile_size=args.tile_size,overlap=args.overlap,amp=args.amp,alpha=args.alpha,jpg_quality=args.jpg_quality,
                   save_color=args.save_color,gt=str(args.gt.resolve()) if args.gt else None,
                   split_sha256=digest(args.split) if args.split else None,version=2)
    state=dict(signature=signature,completed={})
    if args.resume:
        if not manifest_path.exists(): raise FileNotFoundError('No previous meta/run.json')
        state=json.loads(manifest_path.read_text(encoding='utf-8'))
        if state['signature']!=signature: raise ValueError('Resume configuration/weights differ; use another output directory')
    out.mkdir(parents=True,exist_ok=True); (out/'meta').mkdir(exist_ok=True)
    logger=logging.getLogger('nclassinference'); logger.handlers.clear(); logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(),logging.FileHandler(out/'meta/console.log',mode='a' if args.resume else 'w',encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s')); logger.addHandler(handler)
    write_json(out/'meta/classes.json',dict(num_classes=cfg['num_classes'],class_names=cfg['class_names'],palette=cfg['palette'],background_index=cfg['background_index'],ignore_index=cfg['ignore_index']))
    write_json(manifest_path,state)
    logger.info('%s/%s | %d images | %s | %s | parameters=%d',cfg['arch'],cfg['encoder'],len(entries),args.mode,'AMP' if args.amp else 'FP32',sum(p.numel() for p in model.parameters()))
    selected=list(entries.items())[:args.max_images or None]
    try:
        for number,(key,path) in enumerate(selected,1):
            stat=path.stat(); stamp=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
            old=state['completed'].get(key)
            products=['masks/'+key+'.png','overlays/'+key+'_overlay.jpg']
            if args.save_color: products.append('colors/'+key+'_color.png')
            gt_path=args.gt/(key+'.png') if args.gt else None
            gt_hash=digest(gt_path) if gt_path else None
            if old and old['input_stat']==stamp and old.get('gt_sha256')==gt_hash and all((out/p).is_file() for p in products):
                logger.info('[%d/%d] resume skip: %s',number,len(selected),path.name); continue
            with Image.open(path) as im: image=im.convert('RGB')
            target=read_mask(gt_path,image.size,cfg) if gt_path else None
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); start=time.perf_counter()
            mask=predict_image(model,image,cfg,args,torch)
            torch.cuda.synchronize(device); elapsed=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated(device)/1024**3
            save_image(out/products[0],indexed_image(mask,cfg))
            save_image(out/products[1],overlay(image,mask,cfg,args.alpha),'JPEG',quality=args.jpg_quality)
            if args.save_color: save_image(out/products[2],colorize(mask,cfg['palette'],cfg['ignore_index']))
            row=dict(image=path.relative_to(root).as_posix(),width=image.width,height=image.height,seconds=elapsed,peak_gib=peak,
                     input_stat=stamp,gt_sha256=gt_hash,class_pixels=np.bincount(mask.ravel(),minlength=cfg['num_classes']).tolist())
            if target is not None: row['confusion']=confusion(mask,target,cfg['num_classes'],cfg['ignore_index']).tolist()
            state['completed'][key]=row
            write_json(manifest_path,state); write_summary(out,state,entries,cfg)
            logger.info('[%d/%d] %s | %dx%d | %.2fs | peak %.2f GiB',number,len(selected),path.name,image.width,image.height,elapsed,peak)
    except Exception:
        logger.exception('Stopped. Completed results retained; correct the problem and use --resume. OOM does not trigger automatic resize or CPU fallback.')
        raise
    write_summary(out,state,entries,cfg)
    return out


def write_summary(out,state,entries,cfg):
    rows=[]; cm=np.zeros((cfg['num_classes'],cfg['num_classes']),dtype=np.int64)
    for key,info in state['completed'].items():
        if key not in entries: continue
        row={k:info[k] for k in ('image','width','height','seconds','peak_gib')}
        for i,count in enumerate(info['class_pixels']): row[f'class_{i}_pixels']=count
        rows.append(row)
        if 'confusion' in info: cm+=np.asarray(info['confusion'],dtype=np.int64)
    if rows:
        stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        atomic(out/'per_image.csv',lambda f:f.write(stream.getvalue().encode('utf-8-sig')))
    if state['signature']['gt']:
        write_json(out/'metrics.json',dict(**metrics(cm,cfg),images=len(rows),confusion=cm.tolist()))


if __name__=='__main__': run(parse_args())
