from typing import Optional, Union

import numpy as np
import torch
from torch.fx.node import Target
import tensorrt as trt
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor, set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def add_qdq_to_tensor(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    tensor: Union[TRTTensor, torch.Tensor, np.ndarray],
    scale_value: float = 0.1,
    axis: int = 0
) -> TRTTensor:
    """Helper function to add quantization/dequantization to a tensor"""
    if not isinstance(tensor, trt.ITensor):
        tensor = get_trt_tensor(ctx, tensor, f"{name}_tensor_for_qdq")

    # Create scale tensor as constant
    scale_tensor = get_trt_tensor(ctx, np.array([scale_value], dtype=np.float32), f"{name}_scale")

    # Add quantize layer
    quantize_layer = ctx.net.add_quantize(tensor, scale_tensor, trt.DataType.FP8)
    set_layer_name(quantize_layer, target, f"{name}_quantize", source_ir)
    quantize_layer.axis = axis


    # Add dequantize layer
    dequantize_layer = ctx.net.add_dequantize(quantize_layer.get_output(0), scale_tensor)
    set_layer_name(dequantize_layer, target, f"{name}_dequantize", source_ir)
    dequantize_layer.axis = axis

    return dequantize_layer.get_output(0)


def addmm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    mat1: Union[TRTTensor, torch.Tensor, np.ndarray],
    mat2: Union[TRTTensor, torch.Tensor, np.ndarray],
    *,
    beta: Union[float, int],
    alpha: Union[float, int],
) -> TRTTensor:
    print(name)

    # Only apply Q/DQ to mat2 when:
    # 1. mat2 is a torch tensor with fp32 data type
    # 2. mat1 is a TensorRT tensor
    if (isinstance(mat2, torch.Tensor) and mat2.dtype == torch.float32 and
            isinstance(mat1, trt.ITensor)):
        print(f"Applying Q/DQ to mat2 in {name} because mat1 is from dequantize and mat2 is fp32 torch tensor")
        mat2 = add_qdq_to_tensor(ctx, target, source_ir, f"{name}_mat2", mat2)

    mm = impl.matmul.matrix_multiply(ctx, target, source_ir, f"{name}_mm", mat1, mat2)
    if alpha != 1:
        mm = impl.elementwise.mul(
            ctx, target, SourceIR.ATEN, f"{name}_mul_alpha", mm, alpha
        )
    if beta != 1:
        input = impl.elementwise.mul(
            ctx, target, SourceIR.ATEN, f"{name}_mul_beta", input, beta
        )

    return impl.elementwise.add(ctx, target, source_ir, f"{name}_add", input, mm)
