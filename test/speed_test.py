"""
@author: Lowinli
@email: lowinli@outlook.com

"""

import os
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastgpt import (
    CausalLMModelForOnnxGeneration,
)

def evaluate_torch(max_length, num_beams):
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    prompt_text = "Natural language processing (NLP) is the ability of a computer program to understand human language as it is spoken and written"
    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    start = time.time()
    for i in range(10):
        generated_ids = model.generate(
            input_ids,
            max_length=max_length + len(prompt_text),
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
            num_beams=num_beams
        )
    waste_time = time.time() - start
    latency = round(waste_time / 10 * 1000, 3)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    print("=" * 20)
    return latency
    


def evaluate_fastgpt(max_length, num_beams):
    model = CausalLMModelForOnnxGeneration.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    prompt_text = "Natural language processing (NLP) is the ability of a computer program to understand human language as it is spoken and written"
    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    start = time.time()
    for i in range(10):
        generated_ids = model.generate(
            input_ids,
            max_length=max_length + len(prompt_text),
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
            num_beams=num_beams
        )
    waste_time = time.time() - start
    latency = round(waste_time / 10 * 1000, 3)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    print("=" * 20)
    return latency


    
if __name__ == "__main__":
    with open("speed.md", "w") as f:
        f.write("## 生成速度评测(ms)")
        for max_length in [16, 32, 64]:
            f.write(f"+ 生成长度{max_length}评测\n---\n")
            f.write("|模型框架|beam:2|beam:3|beam:4|\n|-|-|-|-|\n|torch|")
            for num_beams in [1, 2, 3]:
                latency = evaluate_torch(max_length, num_beams)
                print(f"torch: max_length{max_length}, num_beam{num_beams}, latency{latency}")
                f.write(f"{latency}|")
            f.write("\n|fastgpt|")
            for num_beams in [1, 2, 3]:
                latency = evaluate_fastgpt(max_length, num_beams)
                print(f"fastgpt: max_length{max_length}, num_beam{num_beams}, latency{latency}")
                f.write(f"{latency}|")
            f.write("\n")
            