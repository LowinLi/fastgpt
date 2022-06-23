"""
@author: Lowinli
@email: lowinli@outlook.com

"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime
import torch
import os
import time
import numpy as np
from torch import tensor
from .model_wrapper import ModelWrapper


def generate_onnx_representation(model, onnx_name_path=None):
    os.makedirs(os.path.join(model.config._name_or_path, "onnx"), exist_ok=True)
    if onnx_name_path is None:
        onnx_name_path = os.path.join(model.config._name_or_path, "onnx", "model.onnx")
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    embed_size_per_head = int(model.config.n_embd / model.config.n_head)
    model_wrapper = ModelWrapper(model)
    model_wrapper.eval()
    past_key_values = torch.randn([n_layer, 2, 1, n_head, 1, embed_size_per_head])
    # past_key_values = torch.randn([n_layer, 2, 1, n_head, int(n_positions/4), embed_size_per_head])
    # 如果解码第一个token，没有past_key_values, 则使用torch.randn([1, 12, 0, 64])，在transformers中等同于past_key_values=None

    test_inputs = {
        "input_ids": torch.LongTensor([[4299, 23748, 62, 6894, 33529]]),
        "attention_mask": torch.LongTensor([[1, 1, 1, 1, 1, 1]]),
        "past_key_values": past_key_values,
    }
    print("onnx dummpy输入配置")
    dummy_inputs = tuple([x for x in test_inputs.values()])
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},  # 动态维度,第0和1维度都是可变的
        "attention_mask": {0: "batch_size", 1: "seq_len"},  # 动态维度,第0和1维度都是可变的
        "past_key_values": {
            2: "batch_size",
            4: "seq_len",
        },  # 层数(不可变), QK(不可变), batch_size(可变), num_heads(不可变), sequence_length(可变), embed_size_per_head(不可变)
    }
    print("onnx转换中...")
    # 导出
    torch.onnx.export(
        model_wrapper,  # 模型
        args=dummy_inputs,  # 模型输入
        f=onnx_name_path,  # 存储路径
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=[
            "input_ids",
            "attention_mask",
            "past_key_values",
        ],
        # 模型输入的名字，随便写
        output_names=["logits", "past_key_values_array"],  # 模型输出的名字，使用ort就要根据这个名字获取结果
        dynamic_axes=dynamic_axes,
    )
    return onnx_name_path


def quantize(onnx_name_path):
    print("onnx量化int8中...")
    model_name = onnx_name_path
    output_model_name = f"{model_name[:-5]}-quantized.onnx"
    quantize_dynamic(
        model_input=model_name,
        model_output=output_model_name,
        per_channel=True,
        reduce_range=True,  # should be the same as per_channel
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,  # per docs, signed is faster on most CPUs
        optimize_model=False,
    )
    return output_model_name


def test_torch_inference(model):
    print("torch加载测试...")
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    embed_size_per_head = int(model.config.n_embd / model.config.n_head)
    model_wrapper = ModelWrapper(model)
    model_wrapper.eval()
    past_key_values = torch.randn([n_layer, 2, 1, n_head, 0, embed_size_per_head])
    # past_key_values = torch.randn([n_layer, 2, 1, n_head, int(n_positions/4), embed_size_per_head])
    # 如果解码第一个token，没有past_key_values, 则使用torch.randn([1, 12, 0, 64])，在transformers中等同于past_key_values=None
    test_inputs = {
        "input_ids": tensor([[4299, 23748, 62, 6894, 33529]]),
        "attention_mask": tensor([[1, 1, 1, 1, 1]]),
        "past_key_values": past_key_values,
    }
    start = time.time()
    with torch.no_grad():
        output = model_wrapper(**test_inputs)
    print("#" * 10 + "\ntorch格式输出：\n")
    print(output[0])
    waste_time = time.time() - start
    latency = round(waste_time * 1000, 3)
    print(f"torch推断用时{latency}ms")
    return output[0]


def test_onnx_inference(onnx_name_path, config):
    print("onnx加载测试...")
    n_layer = config.n_layer
    n_head = config.n_head
    embed_size_per_head = int(config.n_embd / config.n_head)
    session = onnxruntime.InferenceSession(onnx_name_path)
    ort_inputs = {
        "input_ids": np.array([[4299, 23748, 62, 6894, 33529]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
        "past_key_values": np.random.normal(
            size=(n_layer, 2, 1, n_head, 0, embed_size_per_head)
        ).astype(np.float32),
    }
    start = time.time()
    output = session.run(None, ort_inputs)
    print("#" * 10 + "\nonnx格式输出：\n")
    print(output[0])
    waste_time = time.time() - start
    latency = round(waste_time * 1000, 3)
    print(f"onnx推断用时{latency}ms")
    return output[0]


def download_from_transformers(model_name_path):
    print("下载torch模型中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_path)
    tokenizer.save_pretrained(model_name_path)
    model.save_pretrained(model_name_path)
    return tokenizer, model


def transformers_onnx_pipeline(model_name_path="distilgpt2"):
    tokenizer, model = download_from_transformers(model_name_path)
    test_torch_inference(model)
    onnx_path = generate_onnx_representation(model)
    test_onnx_inference(onnx_path, model.config)
    quantized_onnx_path = quantize(onnx_path)
    test_onnx_inference(quantized_onnx_path, model.config)


if __name__ == "__main__":
    transformers_onnx_pipeline("shibing624/code-autocomplete-distilgpt2-python")
