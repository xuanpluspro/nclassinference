"""Image/configuration/output helpers; no torch import."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def atomic(path, writer):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f: writer(f)
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def write_json(path, value):
    atomic(path, lambda f: f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')))

def save_image(path, image, fmt='PNG', **kwargs):
    atomic(path, lambda f: image.save(f, format=fmt, **kwargs))

def images(input_path, split=None):
    root = Path(input_path).resolve()
    if root.is_file():
        if split: raise ValueError('--split requires an image directory')
        found = [root]; root = root.parent
    elif root.is_dir():
        found = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else: raise FileNotFoundError(root)
    if not found: raise ValueError('No input images')
    mapping = {}; case_keys = set()
    for p in found:
        key = p.relative_to(root).with_suffix('').as_posix()
        if key.casefold() in case_keys: raise ValueError(f'Ambiguous output stem: {key}')
        mapping[key] = p; case_keys.add(key.casefold())
    if split:
        ids = [x.strip() for x in Path(split).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        if not ids or len(set(ids)) != len(ids): raise ValueError('Empty or duplicate split IDs')
        selected = {}
        for name in ids:
            # Each line is a complete ID; spaces and Chinese names are preserved.
            if name not in mapping: raise FileNotFoundError(f'Image ID not found: {name}')
            selected[name] = mapping[name]
        mapping = selected
    return root, mapping

def palette(num_classes):
    if not 2 <= num_classes <= 255: raise ValueError('Support 2..255 classes; index 255 reserved for ignore')
    colors = []
    for label in range(num_classes):
        r = g = b = 0; value = label
        for bit in range(8):
            r |= (value & 1) << (7-bit); g |= ((value >> 1) & 1) << (7-bit); b |= ((value >> 2) & 1) << (7-bit); value >>= 3
        colors.append([r,g,b])
    return colors

def validate_schema(cfg):
    n = int(cfg['num_classes']); cfg['num_classes'] = n
    cfg.setdefault('class_names', [f'class_{i}' for i in range(n)])
    cfg.setdefault('background_index', 0); cfg.setdefault('ignore_index', 255)
    cfg.setdefault('palette', palette(n))
    if len(cfg['class_names']) != n or len(set(cfg['class_names'])) != n: raise ValueError('Invalid class names')
    if not 0 <= cfg['background_index'] < n: raise ValueError('Invalid background_index')
    if cfg['ignore_index'] in range(n): raise ValueError('ignore_index overlaps a class ID')
    colors = np.asarray(cfg['palette'])
    if colors.shape != (n,3) or not np.issubdtype(colors.dtype, np.integer) or colors.min() < 0 or colors.max() > 255: raise ValueError('Invalid palette')
    return cfg

def colorize(mask, colors, ignore_index=255):
    mask = np.asarray(mask)
    valid = mask != ignore_index
    if mask.ndim != 2 or np.any((mask[valid] < 0) | (mask[valid] >= len(colors))): raise ValueError('Invalid indexed mask')
    out = np.full((*mask.shape,3), 160, dtype=np.uint8)
    out[valid] = np.asarray(colors, dtype=np.uint8)[mask[valid]]
    return Image.fromarray(out)

def overlay(image, mask, cfg, alpha=.45):
    rgb = np.asarray(image.convert('RGB')).copy(); m = np.asarray(mask)
    if rgb.shape[:2] != m.shape: raise ValueError('Image/mask size mismatch')
    keep = (m != cfg['background_index']) & (m != cfg['ignore_index'])
    color = np.asarray(colorize(m, cfg['palette'], cfg['ignore_index']))
    rgb[keep] = np.rint((1-alpha)*rgb[keep]+alpha*color[keep]).astype(np.uint8)
    return Image.fromarray(rgb)

def read_mask(path, size, cfg):
    with Image.open(path) as im: m = np.asarray(im).copy()
    if m.ndim != 2 or m.shape != (size[1],size[0]): raise ValueError(f'Expected original-size indexed mask: {path}')
    valid = m != cfg['ignore_index']
    if np.any((m[valid]<0)|(m[valid]>=cfg['num_classes'])): raise ValueError(f'Invalid label IDs: {path}')
    return m

def confusion(pred, target, n, ignore=255):
    valid = target != ignore
    return np.bincount((target[valid].astype(np.int64)*n+pred[valid]).ravel(), minlength=n*n).reshape(n,n)

def metrics(cm, cfg):
    cm = np.asarray(cm, dtype=np.float64); tp = cm.diagonal(); gt = cm.sum(1); pred = cm.sum(0); union = gt+pred-tp
    def ratio(a,b): return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)
    iou = ratio(tp,union); dice = ratio(2*tp,gt+pred)
    rows = []
    for i, name in enumerate(cfg['class_names']):
        row = dict(id=i,name=name,iou=iou[i],dice=dice[i],precision=ratio(tp,pred)[i],recall=ratio(tp,gt)[i],support=int(gt[i]))
        rows.append({k: (None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()})
    def mean(values):
        values = values[np.isfinite(values)]; return float(values.mean()) if len(values) else None
    fg = np.arange(len(tp)) != cfg['background_index']
    return dict(valid_pixels=int(cm.sum()),pixel_accuracy=float(tp.sum()/cm.sum()) if cm.sum() else None,
                miou=mean(iou),miou_foreground=mean(iou[fg]),dice=mean(dice),per_class=rows,
                protocol='original-resolution global confusion matrix; mIoU includes background; classes with zero union excluded')

def tile_starts(length, size, overlap):
    if size < 32 or not 0 <= overlap < size: raise ValueError('Require tile_size>=32 and 0<=overlap<tile_size')
    if length <= size: return [0]
    starts = list(range(0,length-size+1,size-overlap))
    if starts[-1] != length-size: starts.append(length-size)
    return starts


def apply_dataset_config(cfg, path):
    """Apply an explicit visualization schema without remapping trained class IDs."""
    supplied = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(supplied, dict): raise ValueError('Dataset config must be a JSON object')
    allowed = {'num_classes', 'class_names', 'background_index', 'ignore_index', 'palette', 'palette_source', 'note', 'legacy_class_names', 'references'}
    if set(supplied) - allowed: raise ValueError('Dataset config must not contain model/architecture settings')
    for key in ('num_classes', 'background_index', 'ignore_index'):
        if key in supplied and supplied[key] != cfg[key]:
            raise ValueError(f'Dataset config conflicts with checkpoint: {key}; class IDs are never remapped')
    names = supplied.get('class_names', cfg['class_names'])
    if names != cfg['class_names'] and cfg['class_names'] not in supplied.get('legacy_class_names', []):
        raise ValueError('Dataset config conflicts with checkpoint: class_names; class IDs are never remapped')
    result = dict(cfg)
    result['class_names'] = names
    if 'palette' in supplied: result['palette'] = supplied['palette']
    return validate_schema(result)


def indexed_image(mask, cfg):
    """P-mode PNG: preserve class IDs exactly, attach the visualization palette."""
    mask = np.asarray(mask)
    valid = mask != cfg['ignore_index']
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer): raise ValueError('Expected integer 2-D mask')
    if np.any((mask[valid] < 0) | (mask[valid] >= cfg['num_classes'])) or np.any((mask < 0) | (mask > 255)):
        raise ValueError('Mask labels cannot be represented in a P-mode PNG')
    # Construct from raw indices: RGB quantization must never change class IDs.
    image = Image.frombytes('P', (mask.shape[1], mask.shape[0]), mask.astype(np.uint8).tobytes())
    colors = [0] * 768
    for i, rgb in enumerate(cfg['palette']): colors[i*3:i*3+3] = list(rgb)
    ignore = cfg['ignore_index']
    if 0 <= ignore < 256: colors[ignore*3:ignore*3+3] = [160,160,160]
    image.putpalette(colors)
    return image



def prediction_products(key, save_color=False, save_overlay=False):
    products = ['masks/'+key+'.png']
    if save_color: products.append('colors/'+key+'_color.png')
    if save_overlay: products.append('overlays/'+key+'_overlay.jpg')
    return products


def save_prediction(out, key, mask, image, cfg, save_color=False, save_overlay=False, alpha=.45, jpg_quality=90):
    """Always write the colored P-mode mask; additional image copies are opt-in."""
    out = Path(out)
    save_image(out/'masks'/(key+'.png'), indexed_image(mask,cfg))
    if save_color: save_image(out/'colors'/(key+'_color.png'), colorize(mask,cfg['palette'],cfg['ignore_index']))
    if save_overlay: save_image(out/'overlays'/(key+'_overlay.jpg'), overlay(image,mask,cfg,alpha), 'JPEG', quality=jpg_quality)
