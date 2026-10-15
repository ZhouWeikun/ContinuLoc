"""Geometry contract tests; run with python -m unittest discover -s tests."""
import unittest
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from trainer_depends.utils.util_data_transform_with_params import (
    GridSamplingQueryGeometry, mk_pil_transform_with_params, apply_augment_to_coords)


class QueryGeometryTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((224, 224))
        self.array = np.stack([x, y, (x + y)//2], axis=-1).astype('uint8')
        self.image = Image.fromarray(self.array)

    def test_identity_and_center(self):
        transform = GridSamplingQueryGeometry(prefilter_resolution_factor=0)
        np.testing.assert_array_equal(np.asarray(transform.render(self.image, 0, 1)), self.array)
        for angle in [0, 37, -90, 180]:
            matrix = transform.inverse_matrix(800, angle, .97, 8)
            np.testing.assert_allclose(matrix @ [112,112,1], [400,400], atol=1e-10)

    def test_rotation_direction(self):
        transform = GridSamplingQueryGeometry(prefilter_resolution_factor=0, pad_mode='zero')
        for angle in [90, -90, 180]:
            actual = np.asarray(transform.render(self.image, angle, 1))
            expected = np.asarray(F.rotate(self.image, -angle, InterpolationMode.BICUBIC))
            # Pillow fast right-angle transpose vs affine can differ by one quantization level.
            self.assertLessEqual(np.abs(actual.astype(float)-expected.astype(float)).max(), 1)

    def test_shear_order(self):
        transform = GridSamplingQueryGeometry(prefilter_resolution_factor=0, pad_mode='zero')
        actual = np.asarray(transform.render(self.image, 0, 1, 8)).astype(float)
        expected = np.asarray(F.affine(self.image, 0, [0,0], 1, [8,0], InterpolationMode.BICUBIC)).astype(float)
        self.assertLess(np.abs(actual-expected).max(), 2)
        # Matrix inverse of forward scale -> rotation -> x-shear.
        a=np.deg2rad(37);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        shear=np.array([[1,-np.tan(np.deg2rad(8))],[0,1]])
        np.testing.assert_allclose(transform.inverse_matrix(224,37,1.05,8)[:,:2] @ (shear@rotation*1.05),np.eye(2),atol=1e-12)

    def test_labels_and_factory(self):
        pipeline=mk_pil_transform_with_params([.5]*3,[.2]*3,rand_rot=True,rotation_range_deg=(37,37),
            rand_scale=True,scale_range=(1.05,1.05),center_crop=True,geometry_method='grid_sampling')
        output=pipeline(self.image)
        self.assertEqual(tuple(output.shape),(3,224,224));self.assertTrue(torch.isfinite(output).all())
        params=pipeline.get_params();self.assertEqual(params['rotation_deg'],37);self.assertEqual(params['scale'],1.05)
        raw=torch.tensor([.2,.3,.5,1.1]);coords=apply_augment_to_coords(raw,params['rotation_deg'],params['scale'])
        torch.testing.assert_close(coords[:2],raw[:2]);self.assertAlmostEqual(float(coords[3]),1.1/1.05,places=6)
        self.assertAlmostEqual(float(coords[2]),.5+np.deg2rad(37),places=6)

    def test_pipelined_default_and_unsupported_geometry(self):
        pipeline = mk_pil_transform_with_params([0]*3, [1]*3, center_crop=True)
        self.assertIsNone(pipeline.grid_sampling_transform)
        expected = F.to_tensor(F.center_crop(F.resize(self.image,224),224))
        torch.testing.assert_close(pipeline(self.image), expected)
        with self.assertRaisesRegex(ValueError, 'extra affine translate'):
            mk_pil_transform_with_params([0]*3,[1]*3,center_crop=True,rand_affine=True,
                affine_para={'translate':(.1,0)},geometry_method='grid_sampling')

    def test_prefilter_and_padding(self):
        y,x=np.indices((1024,1024));checker=Image.fromarray(np.repeat((((x+y)%2)*255).astype('uint8')[...,None],3,axis=2))
        direct=np.asarray(GridSamplingQueryGeometry(prefilter_resolution_factor=0).render(checker,23,1)).astype(float)
        filtered=np.asarray(GridSamplingQueryGeometry(prefilter_resolution_factor=2).render(checker,23,1)).astype(float)
        self.assertLess(filtered[40:-40,40:-40].std(),direct[40:-40,40:-40].std())
        constant=Image.new('RGB',(1000,800),(80,100,120))
        mean_fill=np.asarray(GridSamplingQueryGeometry().render(constant,45,.95))
        self.assertLessEqual(np.max(np.abs(mean_fill.astype(float)-[80,100,120])),1)

if __name__=='__main__': unittest.main()
