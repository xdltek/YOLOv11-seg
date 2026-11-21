"""Device cancellation test for FP32 bias in the generic convolution kernel."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/bin/yolov11_seg_inference_probe"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    weights = np.full((4, 3, 5, 5), .5, np.float32)
    weights[:, -1, -1, -1] = 11.375
    graph = helper.make_graph([
        helper.make_node("Conv", ["images", "weights", "bias"], ["output"],
                         kernel_shape=[5, 5], name="generic_bias_cancellation")],
        "generic-bias", [helper.make_tensor_value_info("images", 1, [1, 3, 6, 66])],
        [helper.make_tensor_value_info("output", 1, [1, 4, 2, 62])],
        [numpy_helper.from_array(weights, "weights"),
         numpy_helper.from_array(np.full(4, -48, np.float32), "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    source = args.output / "generic.onnx"
    onnx.save(model, source)
    inputs = args.output / "input.f32"
    np.ones((1, 3, 6, 66), np.float32).tofile(inputs)
    captures = args.output / "captures"
    subprocess.run([str(args.binary.resolve()), str(source.resolve()), str(inputs.resolve()),
                    str(captures.resolve()), "--application"], check=True, timeout=180)
    # All operands are exact BF16. FP32 MAC is 48.375, bias -48 gives .375.
    # Rounding the accumulator before bias would give .25 or .5 instead.
    bindings = json.loads((captures / "bindings.json").read_text())
    assert len(bindings) == 2
    for binding in bindings:
        actual = np.fromfile(captures / binding["file"], np.float32)
        assert actual.size == 4 * 2 * 62
        np.testing.assert_array_equal(actual, np.full(actual.shape, .375, np.float32))
    print("PASS: generic 5x5 convolution, exact .375 after FP32 bias, two device frames")


if __name__ == "__main__":
    main()
