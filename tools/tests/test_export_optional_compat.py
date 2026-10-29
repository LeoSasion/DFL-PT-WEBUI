"""Optional export packages must work with the pinned production ONNX runtime."""
import importlib.util

import torch  # Load native Torch DLLs before image/export dependency imports on Windows.
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), sess_options=options,
                               providers=['CPUExecutionProvider'])


def test_graphsurgeon_folding_preserves_ir_and_numerical_output():
    gs = pytest.importorskip('onnx_graphsurgeon')
    source = helper.make_model(helper.make_graph([
        helper.make_node('Add', ['a', 'b'], ['sum']),
        helper.make_node('Add', ['x', 'sum'], ['y']),
    ], 'fold', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(np.arange(4, dtype=np.float32), name)
         for name in ['a', 'b']]), opset_imports=[helper.make_opsetid('', 19)], ir_version=10)
    graph = gs.import_onnx(source)
    graph.fold_constants(error_ok=False).cleanup().toposort()
    folded = gs.export_onnx(graph)
    assert len(folded.graph.node) == 1
    assert folded.ir_version == source.ir_version
    onnx.checker.check_model(folded)
    data = np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 4)
    np.testing.assert_array_equal(session(folded).run(None, {'x': data})[0],
                                  session(source).run(None, {'x': data})[0])


def test_modelopt_cpu_int8_quantization_runs_nonzero_output(tmp_path):
    pytest.importorskip('onnx_graphsurgeon')
    if importlib.util.find_spec('modelopt') is None:
        pytest.skip('Optional ModelOpt export dependencies are not installed')
    from modelopt.onnx.quantization import quantize
    weights = np.linspace(-.5, .5, 256, dtype=np.float32).reshape(16, 16)
    source = helper.make_model(helper.make_graph([
        helper.make_node('MatMul', ['x', 'w'], ['dense_out'], name='dense'),
        helper.make_node('Relu', ['dense_out'], ['y']),
    ], 'int8', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 16])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 16])],
        [numpy_helper.from_array(weights, 'w')]),
        opset_imports=[helper.make_opsetid('', 19)], ir_version=10)
    path, output = tmp_path / 'source.onnx', tmp_path / 'int8.onnx'
    onnx.save(source, path)
    values = np.random.default_rng(1029).uniform(-1, 1, (16, 16)).astype(np.float32)
    values[0] = np.linspace(-1, 1, 16, dtype=np.float32)
    values[1] = -values[0]
    quantize(str(path), quantize_mode='int8', calibration_data={'x': values},
             calibration_method='max', calibration_eps=['cpu'],
             nodes_to_quantize=['dense'], op_types_to_quantize=['MatMul'],
             high_precision_dtype='fp32', output_path=str(output),
             simplify=False, autotune=False, dq_only=False,
             enable_gemv_detection_for_trt=False)
    quantized = onnx.load(output)
    onnx.checker.check_model(quantized)
    for kind in ['QuantizeLinear', 'DequantizeLinear']:
        assert any(node.op_type == kind for node in quantized.graph.node)
    expected = session(source).run(None, {'x': values[:2]})[0]
    actual = session(quantized).run(None, {'x': values[:2]})[0]
    assert np.count_nonzero(expected) >= 16 and np.linalg.norm(expected) > 1
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual-expected)) < .05
    assert np.linalg.norm(actual-expected) / np.linalg.norm(expected) < .05
