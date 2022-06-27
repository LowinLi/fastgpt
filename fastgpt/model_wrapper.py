"""
@author: Lowinli
@email: lowinli@outlook.com

"""

import torch


class ModelWrapper(torch.nn.Module):
    r"""
    这是一个onnx格式的装饰类，用于定义确定格式的矩阵输入和矩阵输出，从而做onnx格式转换
    """

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
        )  # 要保证wrapper的输出不能是python的列表，需要是torch矩阵，才能做onnx格式转换
        return res.logits, past_key_values_array
