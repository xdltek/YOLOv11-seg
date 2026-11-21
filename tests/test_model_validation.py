"""Host-only checks for the numerical audit and exact ONNX graph adaptation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2
import onnx
from onnx import helper, numpy_helper

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

    def test_attention_head_and_query_splits_are_equivalent(self):
        graph = helper.make_graph([
            helper.make_node("MatMul",["q","k"],["logits"],name="/attn/MatMul"),
            helper.make_node("Softmax",["logits"],["prob"],axis=-1,name="/attn/Softmax"),
            helper.make_node("MatMul",["v","prob"],["context"],name="/attn/MatMul_1"),
            helper.make_node("Softmax",["dfl"],["distance"],axis=1,name="/dfl/Softmax")],
            "attention-test",[helper.make_tensor_value_info(name,1,shape) for name,shape in (
                ("q",[1,2,6,4]),("k",[1,2,4,6]),("v",[1,2,3,6]),("dfl",[1,16,4,8]))],
            [helper.make_tensor_value_info("context",1,[1,2,3,6]),
             helper.make_tensor_value_info("distance",1,[1,16,4,8])])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)]);model.ir_version=10
        rng=np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as directory:
            src=Path(directory)/"original.onnx";onnx.save(model,src);original=src.read_bytes()
            reference=validation.session(src)
            for stable in (False,True):
                dst=Path(directory)/f"prepared_{stable}.onnx"
                validation.adapt(src,dst,attention_tile=3,split_attention_heads=True,decompose_attention=stable)
                actual=validation.session(dst)
                for _ in range(3):
                    inputs={v.name:rng.normal(size=v.shape).astype(np.float32) for v in reference.get_inputs()}
                    for a,b in zip(reference.run(None,inputs),actual.run(None,inputs)):
                        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
                self.assertEqual(src.read_bytes(),original)

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

    def test_exact_dfl_and_prototype_rewrites(self):
        rng = np.random.default_rng(10)
        weights = rng.normal(size=(4,2,2,2)).astype(np.float32)
        bias = rng.normal(size=2).astype(np.float32)
        graph = helper.make_graph([
            helper.make_node("Softmax",["dfl"],["scores"],axis=1,name="/head/dfl/Softmax"),
            helper.make_node("ConvTranspose",["proto","weight","bias"],["masks"],
                             kernel_shape=[2,2],strides=[2,2],name="/head/proto/upsample")],
            "rewrite-test", [helper.make_tensor_value_info("dfl",1,[1,16,4,8]),
                             helper.make_tensor_value_info("proto",1,[1,4,5,5])],
            [helper.make_tensor_value_info("scores",1,[1,16,4,8]),
             helper.make_tensor_value_info("masks",1,[1,2,10,10])],
            [numpy_helper.from_array(weights,"weight"),numpy_helper.from_array(bias,"bias")])
        model = helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)])
        model.ir_version = 10
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source.onnx"
            onnx.save(model,source)
            original = source.read_bytes()
            reference = validation.session(source)
            for conv_sum in (False,True):
                dest = Path(directory)/f"adapted_{conv_sum}.onnx"
                validation.adapt(source,dest,prototype=True,dfl_conv=conv_sum)
                adapted = validation.session(dest)
                self.assertEqual(source.read_bytes(),original)
                for _ in range(3):
                    inputs = {"dfl":rng.normal(size=(1,16,4,8)).astype(np.float32),
                              "proto":rng.normal(size=(1,4,5,5)).astype(np.float32)}
                    for expected,actual in zip(reference.run(None,inputs),adapted.run(None,inputs)):
                        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)


if __name__ == "__main__":
    unittest.main()
