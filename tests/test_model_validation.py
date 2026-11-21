"""Host-only checks for the original-ONNX numerical audit."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

import validate_models as validation


class ValidationTests(unittest.TestCase):
    def test_integral_center_nearest_rgb_sampling(self):
        # For a 2x reduction, destination (0,0) samples source (1,1), and
        # destination (639,359) samples (1279,719), with 140 padded rows.
        image = np.zeros((720,1280,3),np.uint8)
        image[1,1] = [17,83,251]
        image[719,1279] = [39,191,71]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"integral.png"
            self.assertTrue(cv2.imwrite(str(path),image))
            actual = validation.preprocess_cpu(path,[1,3,640,640])
        np.testing.assert_allclose(actual[0,:,140,0],np.array([251,83,17],np.float32)/255,rtol=0,atol=1e-7)
        np.testing.assert_allclose(actual[0,:,499,639],np.array([71,191,39],np.float32)/255,rtol=0,atol=1e-7)
        self.assertFalse(actual[:,:,:140].any())
        self.assertFalse(actual[:,:,500:].any())


    def test_empty_positive_sample_cannot_pass(self):
        self.assertFalse(validation.compare([], [], .9, .95)["passed"])

    def test_matching_uses_class_and_geometry_not_output_order(self):
        a = dict(cls=0,score=.9,box=[0,0,4,4],mask=np.eye(4,dtype=bool))
        b = dict(cls=1,score=.8,box=[0,0,4,4],mask=np.ones((4,4),bool))
        self.assertTrue(validation.compare([a,b],[b,a],.9,.95)["passed"])
        self.assertFalse(validation.compare([a,b],[a,a],.9,.95)["passed"])
        wrong = dict(a,mask=np.zeros((4,4),bool))
        self.assertFalse(validation.compare([a],[wrong],.9,.95)["passed"])
        wrong_score = dict(a,score=.1)
        self.assertTrue(validation.compare([a],[wrong_score],.9,.95)["passed"])
        self.assertFalse(validation.compare([a],[wrong_score],.9,.95,.05)["passed"])



if __name__ == "__main__":
    unittest.main()
