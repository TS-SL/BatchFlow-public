#!/usr/bin/env python3
"""Generate negative-test ONNX models for the server's tensor-contract checks.

wrong_output_shape.onnx: the graph declares input float32 [N,3,8,8] and output float32 [N,4] (so it passes a
load-time signature check), but reshapes the real [N,4] result to [2N,2] at runtime.
Element count is unchanged, so a check of element count alone cannot detect it.

wrong_input_signature.onnx: input float32 [N,3,4,4] -> output [N,4]. The declared
input signature differs from [N,3,8,8], so the server must refuse to load it.

Weights are constant; these models exist only to exercise shape validation and carry
no accuracy claim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
OUT = HERE / "wrong_output_shape.onnx"
OUT_SIG = HERE / "wrong_input_signature.onnx"


def main() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 4])
    inits = [
        numpy_helper.from_array(np.zeros((192, 4), dtype=np.float32), name="W"),
        numpy_helper.from_array(np.arange(4, dtype=np.float32), name="B"),
        numpy_helper.from_array(np.array([0, 192], dtype=np.int64), name="shape_192"),
        numpy_helper.from_array(np.array([-1, 2], dtype=np.int64), name="shape_m2"),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "shape_192"], ["flat"]),
        helper.make_node("Gemm", ["flat", "W", "B"], ["y4"]),
        helper.make_node("Reshape", ["y4", "shape_m2"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "wrong_output_shape", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, OUT)
    print(f"wrote {OUT.name}")

    x2 = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 4, 4])
    inits2 = [
        numpy_helper.from_array(np.zeros((48, 4), dtype=np.float32), name="W"),
        numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="B"),
        numpy_helper.from_array(np.array([0, 48], dtype=np.int64), name="shape_48"),
    ]
    nodes2 = [
        helper.make_node("Reshape", ["input", "shape_48"], ["flat"]),
        helper.make_node("Gemm", ["flat", "W", "B"], ["output"]),
    ]
    graph2 = helper.make_graph(nodes2, "wrong_input_signature", [x2], [y], inits2)
    model2 = helper.make_model(graph2, opset_imports=[helper.make_opsetid("", 13)])
    model2.ir_version = 8
    onnx.checker.check_model(model2)
    onnx.save(model2, OUT_SIG)
    print(f"wrote {OUT_SIG.name}")


if __name__ == "__main__":
    main()
