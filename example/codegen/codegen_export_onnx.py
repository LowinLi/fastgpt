"""
LowinLi

https://github.com/salesforce/CodeGen#demo
"""

import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
codegen_dir = os.path.join(file_dir, "CodeGen")
sys.path.append(codegen_dir)
from CodeGen.jaxformer.hf.sample import (
    create_model,
    create_custom_gpt2_tokenizer,
    set_seed,
)

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(file_dir)), "fastgpt"))
from onnx_exporter import (
    generate_onnx_representation,
    quantize,
    test_onnx_inference,
    test_torch_inference,
)

ckpt = "checkpoints/codegen-350M-multi"
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


# completion = sample(device=device, model=model, tokenizer=tokenizer, context=context, pad_token_id=pad, num_return_sequences=batch_size, temp=t, top_p=p, max_length_sample=max_length)[0]
# truncation = truncate(completion)

# print('=' * 100)
# print(completion)
# print('=' * 100)
# print(context+truncation)
# print('=' * 100)

# 导出onnx
test_torch_inference(model)
# onnx_path = generate_onnx_representation(model)
model_path = "checkpoints/codegen-350M-multi"
onnx_path = os.path.join(model_path, "onnx/model.onnx")

test_onnx_inference(onnx_path, model.config)
# 量化onnx
quantized_onnx_path = quantize(onnx_path)

onnx_path = generate_onnx_representation(model)

test_onnx_inference(quantized_onnx_path, model.config)
