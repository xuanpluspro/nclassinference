import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from common import images, palette, validate_schema, colorize, overlay, read_mask, confusion, metrics, tile_starts, write_json, digest
from predict import parse_args
from compare import main as compare


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.cfg=validate_schema(dict(num_classes=2,class_names=['background','leaf']))
    def tearDown(self): self.tmp.cleanup()
    def image(self,key='图 像',size=(37,29)):
        p=self.root/'images'/(key+'.jpg'); p.parent.mkdir(parents=True,exist_ok=True); Image.new('RGB',size,'white').save(p); return p
    def test_spaces_split(self):
        self.image(); split=self.root/'test.txt'; split.write_text('图 像\n',encoding='utf-8-sig')
        _,entries=images(self.root/'images',split); self.assertEqual(list(entries),['图 像'])
    def test_collision(self):
        p=self.image(); Image.new('RGB',(37,29)).save(p.with_suffix('.png'))
        with self.assertRaises(ValueError): images(self.root/'images')
    def test_palette(self):
        self.assertEqual(palette(4),[[0,0,0],[128,0,0],[0,128,0],[128,128,0]])
    def test_schema_rejects(self):
        with self.assertRaises(ValueError): validate_schema(dict(num_classes=2,class_names=['x','x']))
    def test_color_ignore(self):
        m=np.array([[0,1,255]],dtype=np.uint8); c=np.asarray(colorize(m,self.cfg['palette']))
        np.testing.assert_array_equal(c[0,2],[160,160,160])
    def test_overlay_keeps_background(self):
        rgb=Image.new('RGB',(2,1),'white'); m=np.array([[0,1]],dtype=np.uint8)
        a=np.asarray(overlay(rgb,m,self.cfg,1)); np.testing.assert_array_equal(a[0,0],[255]*3); np.testing.assert_array_equal(a[0,1],[128,0,0])
    def test_indexed_gt(self):
        p=self.root/'mask.png'; im=Image.fromarray(np.array([[0,1]],dtype=np.uint8),'P'); im.save(p)
        self.assertEqual(read_mask(p,(2,1),self.cfg).tolist(),[[0,1]])
        with self.assertRaises(ValueError): read_mask(p,(1,2),self.cfg)
    def test_global_metrics(self):
        gt=np.array([[0,1,1,255]],dtype=np.uint8); pred=np.array([[0,0,1,1]],dtype=np.uint8)
        cm=confusion(pred,gt,2); self.assertEqual(cm.tolist(),[[1,0],[1,1]])
        m=metrics(cm,self.cfg); self.assertEqual(m['miou'],.5); self.assertEqual(m['valid_pixels'],3)
    def test_absent_class(self):
        m=metrics(np.array([[2,0],[0,0]]),self.cfg)
        self.assertEqual(m['miou'],1); self.assertIsNone(m['per_class'][1]['iou'])
        self.assertIsNone(metrics(np.zeros((2,2)),self.cfg)['miou'])
    def test_tile_coverage(self):
        for length in (17,512,513,1004,3072):
            cover=np.zeros(length)
            for start in tile_starts(length,512,128): cover[start:min(length,start+512)]+=1
            self.assertTrue((cover>0).all())
    def test_atomic_json(self):
        p=self.root/'目录/a.json'; write_json(p,dict(x='中文')); self.assertEqual(json.loads(p.read_text())['x'],'中文'); self.assertEqual(len(digest(p)),64)
    def test_defaults(self):
        a=parse_args(['--weights','a.pth','--input','imgs','--output','out'])
        self.assertEqual(a.mode,'resize'); self.assertFalse(a.amp)
    def test_compare_panels(self):
        self.image(); run=self.root/'run'; write_json(run/'meta/classes.json',self.cfg)
        (run/'masks').mkdir(); Image.fromarray(np.zeros((29,37),np.uint8)).save(run/'masks/图 像.png')
        output=self.root/'panels'; compare(['--input',str(self.root/'images'),'--runs','UNet='+str(run),'--output',str(output),'--panel-size','128','--font-size','12'])
        with Image.open(output/'图 像_comparison.png') as im: self.assertEqual(im.size,(256,160))
        self.assertTrue((output/'legend.png').is_file())
    def test_compare_schema_mismatch(self):
        self.image()
        for label,cls in [('one',['background','leaf']),('two',['background','stem'])]: write_json(self.root/label/'meta/classes.json',validate_schema(dict(num_classes=2,class_names=cls)))
        with self.assertRaises(ValueError): compare(['--input',str(self.root/'images'),'--runs','A='+str(self.root/'one'),'B='+str(self.root/'two'),'--output',str(self.root/'out')])

if __name__=='__main__': unittest.main()
