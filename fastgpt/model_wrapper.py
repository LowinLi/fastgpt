"""
@author: Lowinli
@email: lowinli@outlook.com

"""

import torch


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values,
    ):
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values_array = torch.stack(
            [torch.stack(x) for x in res.past_key_values]
        )
        return res.logits, past_key_values_array
