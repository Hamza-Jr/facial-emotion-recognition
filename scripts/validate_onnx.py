import onnx

from src.config import ONNX_MODEL_PATH


if not ONNX_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {ONNX_MODEL_PATH}"
    )

print(f"Loading ONNX model: {ONNX_MODEL_PATH}")

model = onnx.load(ONNX_MODEL_PATH)

print("Checking ONNX model...")
onnx.checker.check_model(model)

print("ONNX model is valid.")

input_tensor = model.graph.input[0]
output_tensor = model.graph.output[0]

print(f"Input name: {input_tensor.name}")
print(f"Output name: {output_tensor.name}")

input_shape = [
    dimension.dim_value
    for dimension in input_tensor.type.tensor_type.shape.dim
]

output_shape = [
    dimension.dim_value
    for dimension in output_tensor.type.tensor_type.shape.dim
]

input_dtype = input_tensor.type.tensor_type.elem_type
output_dtype = output_tensor.type.tensor_type.elem_type

print(f"Input shape: {input_shape}")
print(f"Input dtype: {input_dtype}")
print(f"Output shape: {output_shape}")
print(f"Output dtype: {output_dtype}")
