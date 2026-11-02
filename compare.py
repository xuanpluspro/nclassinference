"""Aligned original/GT/model panels from indexed masks. No torch or GPU needed."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import images, read_mask, colorize, overlay, save_image, write_json, validate_schema


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--runs',required=True,nargs='+',help='Label=/path/to/inference/output')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gt',type=Path); p.add_argument('--split',type=Path)
    p.add_argument('--view',choices=['color','overlay'],default='color')
    p.add_argument('--alpha',type=float,default=.45)
    p.add_argument('--panel-size',type=int,default=512)
    p.add_argument('--columns',type=int,default=4)
    p.add_argument('--font',type=Path,help='TTF/OTF with Chinese glyphs if labels use Chinese')
    p.add_argument('--font-size',type=int,default=22)
    p.add_argument('--crop',type=int,nargs=4,metavar=('X','Y','W','H'))
    p.add_argument('--format',choices=['png','jpg'],default='png')
    p.add_argument('--max-images',type=int,default=0)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(argv)
    if a.panel_size<64 or a.columns<1 or a.font_size<8 or a.max_images<0 or not 0<=a.alpha<=1: p.error('Invalid layout options')
    if a.crop and (min(a.crop[:2])<0 or min(a.crop[2:])<1): p.error('Invalid crop')
    root,entries=images(a.input,a.split); runs=[]; schema=None; seen=set()
    for item in a.runs:
        if '=' not in item: p.error('Expected Label=directory')
        label,path=item.split('=',1); path=Path(path).resolve()
        if not label or label in seen: p.error('Unique nonempty labels required')
        seen.add(label)
        cfg=validate_schema(json.loads((path/'meta/classes.json').read_text(encoding='utf-8')))
        signature={k:cfg[k] for k in ('num_classes','class_names','palette','background_index','ignore_index')}
        if schema is not None and schema!=signature: raise ValueError('Class IDs/names/palette differ between models')
        schema=signature; runs.append((label,path))
    protocols = []
    for label, path in runs:
        manifest = path/'meta/run.json'
        if manifest.is_file():
            sig=json.loads(manifest.read_text(encoding='utf-8'))['signature']
            effective_size=(sig.get('size') or sig.get('config',{}).get('size',512)) if sig['mode']=='resize' else None
            protocols.append((label,(sig['mode'],effective_size,sig['amp'],sig.get('tile_size') if sig['mode']=='tile' else None,sig.get('overlap') if sig['mode']=='tile' else None)))
        else:
            print(f'Notice: {label} uses external masks; verify its inference scale/precision/postprocessing manually.',flush=True)
    if len({item[1] for item in protocols})>1:
        print('Warning: model inference modes/sizes/precision differ. This is not a controlled same-protocol comparison.',flush=True)
    out=a.output.resolve()
    for protected in [root]+[r[1] for r in runs]+([a.gt.resolve()] if a.gt else []):
        if out==protected or protected in out.parents: raise ValueError('Comparison output must be outside image/GT/run directories')
    if out.exists() and any(out.iterdir()) and not a.overwrite: raise FileExistsError('Use new output or --overwrite')
    # Validate every requested mask before writing panels, never substitute blank predictions.
    chosen=list(entries.items())[:a.max_images or None]
    for key,_ in chosen:
        for _,r in runs:
            if not (r/'masks'/(key+'.png')).is_file(): raise FileNotFoundError(r/'masks'/(key+'.png'))
        if a.gt and not (a.gt/(key+'.png')).is_file(): raise FileNotFoundError(a.gt/(key+'.png'))
    if a.font: font=ImageFont.truetype(str(a.font),a.font_size)
    else:
        try: font=ImageFont.truetype('DejaVuSans.ttf',a.font_size)
        except OSError: font=ImageFont.load_default()
    write_json(out/'comparison.json',dict(input=str(root),runs=[dict(label=l,path=str(r)) for l,r in runs],gt=str(a.gt) if a.gt else None,view=a.view,crop=a.crop,panel_size=a.panel_size,classes=schema))
    for i,(key,path) in enumerate(chosen,1):
        with Image.open(path) as im: original=im.convert('RGB')
        panels=[('Original',original)]
        def render(m): return colorize(m,schema['palette'],schema['ignore_index']) if a.view=='color' else overlay(original,m,schema,a.alpha)
        if a.gt: panels.append(('GT',render(read_mask(a.gt/(key+'.png'),original.size,schema))))
        for label,r in runs: panels.append((label,render(read_mask(r/'masks'/(key+'.png'),original.size,schema))))
        ncols=min(a.columns,len(panels)); nrows=math.ceil(len(panels)/ncols)
        header=a.font_size+20; cell=a.panel_size
        canvas=Image.new('RGB',(ncols*cell,nrows*(cell+header)), 'white'); draw=ImageDraw.Draw(canvas)
        for j,(label,im) in enumerate(panels):
            if a.crop:
                x,y,w,h=a.crop
                if x+w>im.width or y+h>im.height: raise ValueError(f'Crop exceeds image: {key}')
                im=im.crop((x,y,x+w,y+h))
            # Prediction colors are categorical; nearest display scaling avoids new class colors.
            im=im.copy(); im.thumbnail((cell,cell),Image.Resampling.NEAREST if a.view=='color' and j else Image.Resampling.LANCZOS)
            x=(j%ncols)*cell; y=(j//ncols)*(cell+header)
            if draw.textbbox((0,0),label,font=font)[2] > cell-12: raise ValueError('Label too wide; shorten label or enlarge --panel-size')
            draw.text((x+8,y+6),label,fill='black',font=font)
            canvas.paste(im,(x+(cell-im.width)//2,y+header+(cell-im.height)//2))
        target=out/(key+'_comparison.'+a.format)
        save_image(target,canvas,'PNG' if a.format=='png' else 'JPEG',**({'quality':95} if a.format=='jpg' else {}))
        print(f'[{i}/{len(chosen)}] {target}',flush=True)
    # Standalone class legend, identical to every model and GT.
    legend=Image.new('RGB',(max(400,a.panel_size),len(schema['class_names'])*(a.font_size+16)), 'white'); d=ImageDraw.Draw(legend)
    for i,(name,color) in enumerate(zip(schema['class_names'],schema['palette'])):
        y=i*(a.font_size+16); d.rectangle((8,y+5,30,y+27),fill=tuple(color)); d.text((42,y+3),f'{i}: {name}',font=font,fill='black')
    save_image(out/'legend.png',legend)

if __name__=='__main__': main()
