"""
@author: Lowinli
@email: lowinli@outlook.com

"""

import os
import sys
import unittest
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastgpt import (
    CausalLMModelForOnnxGeneration,
)


class Tests(unittest.TestCase):
    def test_model(self):
        model = CausalLMModelForOnnxGeneration.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

        prompt_text = "Natural language processing (NLP) is the ability of a computer program to understand human language as it is spoken and written"
        input_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
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
