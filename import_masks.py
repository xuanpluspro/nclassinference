"""Normalize external model/ablation masks for compare.py; no model execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from common import images, validate_schema, read_mask, save_image, write_json, colorize, overlay, indexed_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path,help='Original image directory')
    p.add_argument('--masks',required=True,type=Path)
    p.add_argument('--class-config',required=True,type=Path,help='Use another run meta/classes.json to share exact schema')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--suffix',default='',help='Mask stem suffix, e.g. _mask')
    p.add_argument('--binary-255',action='store_true',help='Explicitly convert binary 0/255 foreground encoding to 0/1')
    p.add_argument('--split',type=Path)
    a=p.parse_args()
    cfg=validate_schema(json.loads(a.class_config.read_text(encoding='utf-8-sig')))
    if a.binary_255 and cfg['num_classes']!=2: p.error('--binary-255 requires two classes')
    root,entries=images(a.input,a.split); out=a.output.resolve()
    for protected in (root,a.masks.resolve()):
        if out==protected or protected in out.parents: raise ValueError('Output must be outside original/mask directories')
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a new or empty output')
    # Preflight all paths before writing any output.
    for key in entries:
        if not (a.masks/(key+a.suffix+'.png')).is_file(): raise FileNotFoundError(a.masks/(key+a.suffix+'.png'))
    for key,path in entries.items():
        with Image.open(path) as im: rgb=im.convert('RGB')
        source=a.masks/(key+a.suffix+'.png')
        if a.binary_255:
            with Image.open(source) as im: m=np.asarray(im).copy()
            if m.ndim!=2 or m.shape!=(rgb.height,rgb.width) or not set(np.unique(m)).issubset({0,255}): raise ValueError(f'Not original-size binary 0/255 mask: {source}')
            m=(m==255).astype(np.uint8)
        else: m=read_mask(source,rgb.size,cfg)
        save_image(out/'masks'/(key+'.png'),indexed_image(m,cfg))
        save_image(out/'colors'/(key+'_color.png'),colorize(m,cfg['palette'],cfg['ignore_index']))
        save_image(out/'overlays'/(key+'_overlay.jpg'),overlay(rgb,m,cfg),'JPEG',quality=90)
    write_json(out/'meta/classes.json',cfg)
    write_json(out/'meta/import.json',dict(source=str(a.masks.resolve()),binary_255=a.binary_255,suffix=a.suffix,images=len(entries),note='External predictions; original inference settings must be documented separately'))
    print(f'Imported {len(entries)} masks into {out}')

if __name__=='__main__': main()
