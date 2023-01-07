from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.util.to_channels_last import to_channels_last
import hls4ml


onnx_model = ModelWrapper("./model/CNN/mnist_CNN_no_bias_clean_channels_last_clean.onnx")
onnx_model = onnx_model.transform(GemmToMatMul())
lonnx_model = cleanup_model(onnx_model)
onnx_model.save("./model/CNN/mnist_CNN_no_bias_clean_channels_last_clean.onnx")

model_name = "./model/CNN/mnist_CNN_no_bias_clean_channels_last_clean.onnx"

config = hls4ml.utils.config_from_onnx_model(model_name, granularity='model')
hls_model = hls4ml.converters.convert_from_onnx_model(model_name, hls_config=config)
hls_model.compile()