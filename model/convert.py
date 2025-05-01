import onnx
from onnxconverter_common import float16

model = onnx.load("model/best.onnx")              # load your FP32 ONNX model
model_fp16 = float16.convert_float_to_float16(model) # convert to FP16
onnx.save(model_fp16, "model/best_light.onnx")     # save the FP16 model
