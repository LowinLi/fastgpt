"""
LowinLi

https://github.com/salesforce/CodeGen#demo
"""

import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
codegen_dir = os.path.join(file_dir, "CodeGen")
sys.path.append(codegen_dir)
from jaxformer.hf.sample import (
    create_model,
    create_custom_gpt2_tokenizer,
    set_seed,
)

from fastgpt import (
    generate_onnx_representation,
    quantize,
    test_onnx_inference,
    test_torch_inference,
)


def export(ckpt="checkpoints/codegen-350M-mono"):
    device = "cpu"
    pad = 50256
    model = create_model(ckpt=ckpt, fp16=False)
    model.eval()
    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = pad
    # @markdown # Try out the model
    rng_seed = 42  # @param {type:"integer"}
    rng_deterministic = True  # @param {type:"boolean"}
    p = 0.95  # @param {type:"number"}
    t = 0.2  # @param {type:"number"}
    max_length = 128  # @param {type:"integer"}
    batch_size = 1  # @param {type:"integer"}
    context = "def hello_world():"  # @param {type:"string"}
    set_seed(rng_seed, deterministic=rng_deterministic)

    # 测试torch输出
    test_torch_inference(model)

    # 转换onnx
    onnx_path = generate_onnx_representation(model)
    model_path = ckpt
    onnx_path = os.path.join(model_path, "onnx/model.onnx")
    test_onnx_inference(onnx_path, model.config)

    # 量化onnx
    quantized_onnx_path = quantize(onnx_path)
    test_onnx_inference(quantized_onnx_path, model.config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model_path")
    args = parser.parse_args()
    export(args.model_path)
