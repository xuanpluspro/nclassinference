import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from common import validate_schema, apply_dataset_config, save_prediction, prediction_products
from predict import parse_args

ROOT=Path(__file__).resolve().parents[1]

class GWFSSOutputTests(unittest.TestCase):
    def setUp(self):
        self.path=ROOT/'configs/gwfss.json'
        self.cfg=apply_dataset_config(validate_schema(dict(num_classes=4,class_names=['background','class_1','class_2','class_3'])),self.path)
    def test_names_and_local_palette(self):
        self.assertEqual(self.cfg['class_names'],['background','head','stem','leaf'])
        self.assertEqual(self.cfg['palette'],[[0,0,0],[50,255,132],[50,132,255],[214,255,50]])
    def test_canonical_names_accepted(self):
        cfg=validate_schema(dict(num_classes=4,class_names=['background','head','stem','leaf']))
        self.assertEqual(apply_dataset_config(cfg,self.path)['class_names'],cfg['class_names'])
    def test_wrong_order_rejected(self):
        cfg=validate_schema(dict(num_classes=4,class_names=['background','leaf','stem','head']))
        with self.assertRaisesRegex(ValueError,'class_names'): apply_dataset_config(cfg,self.path)
    def test_default_only_mask_and_exact_colors(self):
        with tempfile.TemporaryDirectory() as d:
            mask=np.arange(4,dtype=np.uint8)[None,:]
            save_prediction(d,'图 像',mask,Image.new('RGB',(4,1)),self.cfg)
            self.assertEqual(sorted(p.relative_to(d).as_posix() for p in Path(d).rglob('*') if p.is_file()),['masks/图 像.png'])
            with Image.open(Path(d)/'masks/图 像.png') as im:
                self.assertEqual(im.mode,'P');np.testing.assert_array_equal(np.asarray(im),mask)
                np.testing.assert_array_equal(np.asarray(im.convert('RGB'))[0],self.cfg['palette'])
    def test_optional_color_without_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            save_prediction(d,'a',np.zeros((2,2),np.uint8),Image.new('RGB',(2,2)),self.cfg,save_color=True)
            self.assertTrue((Path(d)/'colors/a_color.png').is_file());self.assertFalse((Path(d)/'overlays').exists())
    def test_optional_overlay_without_color(self):
        with tempfile.TemporaryDirectory() as d:
            save_prediction(d,'a',np.ones((2,2),np.uint8),Image.new('RGB',(2,2)),self.cfg,save_overlay=True)
            self.assertTrue((Path(d)/'overlays/a_overlay.jpg').is_file());self.assertFalse((Path(d)/'colors').exists())
    def test_products_match_requested_outputs(self):
        self.assertEqual(prediction_products('nested/a'),['masks/nested/a.png'])
        self.assertEqual(prediction_products('a',False,True),['masks/a.png','overlays/a_overlay.jpg'])
    def test_cli_mask_only_default(self):
        a=parse_args(['--weights','a','--input','b','--output','c'])
        self.assertFalse(a.save_color);self.assertFalse(a.save_overlay)

if __name__=='__main__': unittest.main()
