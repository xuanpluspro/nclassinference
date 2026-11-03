import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from common import apply_dataset_config, validate_schema, colorize, indexed_image, save_image
from predict import parse_args
from compare import main as compare

ROOT=Path(__file__).resolve().parents[1]
RGB=[[0,0,0],[0,180,70],[255,255,0],[255,0,0],[0,120,210],[128,40,170]]

class RiceSegPaletteTests(unittest.TestCase):
    def setUp(self):
        self.config=ROOT/'configs/riceseg.json'
        self.base=json.loads(self.config.read_text())
        self.base.pop('palette'); self.base.pop('palette_source')
        self.base=validate_schema(self.base)
        self.cfg=apply_dataset_config(self.base,self.config)
    def test_config_exact_colors(self):
        self.assertEqual(self.cfg['palette'],RGB)
        self.assertNotEqual(self.base['palette'],RGB) # source config was not mutated
    def test_colorize_all_ids(self):
        mask=np.arange(6,dtype=np.uint8)[None,:]
        np.testing.assert_array_equal(np.asarray(colorize(mask,self.cfg['palette']))[0],RGB)
    def test_p_mode_roundtrip_preserves_indices_and_palette(self):
        mask=np.array([[0,1,2],[3,4,5],[255,0,1]],dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'indices.png';save_image(path,indexed_image(mask,self.cfg))
            with Image.open(path) as im:
                self.assertEqual(im.mode,'P');np.testing.assert_array_equal(np.asarray(im),mask)
                self.assertEqual(im.getpalette()[:18],sum(RGB,[]))
                self.assertEqual(im.getpalette()[765:768],[160]*3)
    def test_incompatible_class_count_rejected(self):
        with self.assertRaisesRegex(ValueError,'num_classes'):
            apply_dataset_config(validate_schema(dict(num_classes=2)),self.config)
    def test_wrong_class_order_rejected(self):
        wrong=dict(self.base);wrong['class_names']=list(reversed(wrong['class_names']))
        with self.assertRaisesRegex(ValueError,'class_names'): apply_dataset_config(wrong,self.config)
    def test_cli_accepts_dataset_config(self):
        a=parse_args(['--weights','a.pth','--input','imgs','--output','out','--dataset-config',str(self.config)])
        self.assertEqual(a.dataset_config,self.config)
    def test_gt_prediction_panels_have_exact_same_colors(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'images').mkdir();(root/'gt').mkdir();(root/'run/masks').mkdir(parents=True)
            Image.new('RGB',(128,128),'white').save(root/'images/a.png')
            mask=np.full((128,128),4,dtype=np.uint8)
            save_image(root/'gt/a.png',indexed_image(mask,self.cfg));save_image(root/'run/masks/a.png',indexed_image(mask,self.cfg))
            (root/'run/meta').mkdir();(root/'run/meta/classes.json').write_text(json.dumps(self.cfg))
            compare(['--input',str(root/'images'),'--gt',str(root/'gt'),'--runs','Model='+str(root/'run'),'--output',str(root/'figures'),'--panel-size','128','--font-size','12'])
            with Image.open(root/'figures/a_comparison.png') as im:
                self.assertEqual(im.getpixel((128+64,32+64)),tuple(RGB[4]))
                self.assertEqual(im.getpixel((256+64,32+64)),tuple(RGB[4]))

if __name__=='__main__': unittest.main()
