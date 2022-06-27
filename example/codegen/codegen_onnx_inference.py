"""
LowinLi

https://github.com/salesforce/CodeGen#demo
"""

import sys
import os
from tqdm import tqdm
from fastgpt import CausalLMModelForOnnxGeneration

file_dir = os.path.dirname(os.path.abspath(__file__))
codegen_dir = os.path.join(file_dir, "CodeGen")
sys.path.append(codegen_dir)
from CodeGen.jaxformer.hf.sample import create_custom_gpt2_tokenizer
from CodeGen.jaxformer.hf.codegen.configuration_codegen import CodeGenConfig


def get_codegen_model_tokenizer(model_name="codegen-350M-multi", threads=1):
    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = "left"
    pad = 50256
    tokenizer.pad_token = pad
    model_path = os.path.join("checkpoints", model_name)
    onnx_model_path = os.path.join(model_path, "onnx/model-quantized.onnx")
    config = CodeGenConfig.from_pretrained(model_path)
    model = CausalLMModelForOnnxGeneration(onnx_model_path, model_path, config, threads)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = get_codegen_model_tokenizer()
    prompts = [
        """from torch import nn
        class LSTM(Module):
            def __init__(self, *,
                        n_tokens: int,
                        embedding_size: int,
                        hidden_size: int,
                        n_layers: int):""",
        """import numpy as np
        import torch
        import torch.nn as""",
        "import java.util.ArrayList",
        "def factorial(n):",
    ]
    for prompt in tqdm(prompts):
        input_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
        generated_ids = model.generate(
            input_ids,
            max_length=64 + input_ids.shape[1],
            decoder_start_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            output_scores=True,
            temperature=1,
            repetition_penalty=1.0,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            length_penalty=2.0,
            early_stopping=True,
        )
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        print("=" * 20)
