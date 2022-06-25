# fastgpt

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastgpt.svg)](https://pypi.org/project/fastgpt/)
[![PyPI](https://img.shields.io/pypi/v/fastgpt.svg)](https://pypi.org/project/fastgpt/)
[![PyPI](https://img.shields.io/pypi/dw/fastgpt)](https://pypi.org/project/fastgpt/#description)
[![GitHub license badge](https://img.shields.io/github/license/LowinLi/fastgpt)](https://github.com/LowinLi/fastgpt/blob/main/LICENSE)
[![Blog](https://img.shields.io/badge/blog-LowinLi-important)](https://lowin.li)
![Codecov](https://img.shields.io/codecov/c/github/LowinLi/fastgpt)

## fastgpt 是什么

- [fastgpt](https://github.com/LowinLi/fastgpt)是一个基于[transformers](https://github.com/huggingface/transformers)和[onnxruntime](https://github.com/microsoft/onnxruntime)的**python**库，可以无缝衔接的使用 onnxruntime 量化后的 transfromers GPT 模型做文本生成任务，提高推理速度、降低资源成本。

## fastgpt 的背景

- **GPT**模型是通过序列文本预测下一个词的训练任务得到的预训练模型，可以在文本生成任务上达到很好的效果。
- **transformers**库是近些年最火的做预训练模型的 python 库，在其背后的社区，网友、组织分享开源了各式各样的预训练模型，尤其是截止 2022 年 6 月 23 日，[社区](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)的开源文本生成模型多达到**5068**个。
- **onnx**是由微软，亚马逊 ，Facebook 和 IBM 等公司共同开发的，针对机器学习所设计的开放式的文件格式，经过 onnxruntime 量化压缩的预训练模型，在 cpu 硬件上推理速度在各开源框架的对比中首屈一指。

* 然而，通过**transformers**官方的 onnx 接口转换、onnx 量化 API，却没有做好 GPT 模型转换的兼容问题，经常转换失败。而手动进行 onnx 转换需要自定义很多配置，对于新手不很友好。

- **fastgpt**库，就是为了无缝衔接 transformers 库调用 GPT 模型转换 onnx 格式推理，使用者可以在仅修改两行代码的情况下，使用 onnx 量化后的GPT模型，做transformers库的文本生成函数。

* 原 transformers 代码：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```

- fastgpt 代码：

```python
from fastgpt import CausalLMModelForOnnxGeneration
model = CausalLMModelForOnnxGeneration.from_pretrained("distilgpt2")
```

- 在 fastgpt 这一行代码中，会执行以下流程
  1. transformers hub 的模型下载
  2. pytorch 模型推理，输出 logits
  3. onnx 格式转换
  4. onnx 格式模型推理，输出 logits，进行对比差异
  5. onnx 量化
  6. onnx 量化格式模型推理，输出 logits，进行对比差异
  7. 把兼容transformers文本生成函数的onnx格式GPT模型，包装到model中

## 安装

```bash
pip install fastgpt
```

## 快速 demo

```python
from transformers import AutoTokenizer
from fastgpt import CausalLMModelForOnnxGeneration
model = CausalLMModelForOnnxGeneration.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

prompt_text = "Natural language processing (NLP) is the ability of a computer program to understand human language as it is spoken and written"
input_ids = tokenizer(
    prompt_text, return_tensors="pt", add_special_tokens=False
).input_ids

generated_ids = model.generate(   # 这里完全兼容transformers的generate函数
    input_ids,
    max_length=64 + len(prompt_text),
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
```

## fastgpt 的优点

1. **兼容 transformers**: 基于 transformers 库的[文本生成函数](https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/generation_utils.py#L845)，功能非常丰富。fastgpt 在 onnx 格式模型上，兼容该函数。
2. **兼容 cache**: 在文本生成的一个个 token 生成过程中的`past_key_value`需要在 GPT 模型上持续迭代输入，fastgpt 已经通过 onnx 格式做好衔接。
3. **代码修改低成本**：代码替换原版 transformers 仅需修改两行代码。
4. **onnx 格式占内存小**：对于 distilgpt2 模型，torch 版`318MB`, onnx 量化版`243MB`
5. **cpu 上速度更快**: 用时约降低 **33%**

## 生成速度评测(ms)

+ 生成长度4评测

|模型框架|beam:1|beam:2|beam:3|beam:4|
|-|-|-|-|-|
|torch|290.779|475.693|560.458|648.756|
|fastgpt|195.265|292.272|378.933|466.14|
---
+ 生成长度8评测

|模型框架|beam:1|beam:2|beam:3|beam:4|
|-|-|-|-|-|
|torch|482.199|817.065|905.646|1052.983|
|fastgpt|341.735|471.028|583.264|713.009|
---
+ 生成长度16评测

|模型框架|beam:1|beam:2|beam:3|beam:4|
|-|-|-|-|-|
|torch|878.338|1518.198|1619.336|1813.197|
|fastgpt|635.157|838.787|1009.497|1210.047|
---
+ 生成长度32评测

|模型框架|beam:1|beam:2|beam:3|beam:4|
|-|-|-|-|-|
|torch|1661.819|2854.889|3081.585|3436.284|
|fastgpt|1238.585|1599.724|1921.785|2256.674|
---
+ 生成长度64评测

|模型框架|beam:1|beam:2|beam:3|beam:4|
|-|-|-|-|-|
|torch|3257.929|4274.201|4256.85|4677.168|
|fastgpt|2510.484|3081.851|2697.296|3150.157|
---
model name	: Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz
cpu cores	: 2

详见GITHUB ACTIONS的[cml报告](https://github.com/LowinLi/fastgpt/commit/044567f960dd05fb0ef06870cf12002718ed5861#commitcomment-76805953)

## 感谢

- [transformers](https://github.com/huggingface/transformers)
- [fastT5](https://github.com/Ki6an/fastT5)
- [onnxruntime](https://github.com/microsoft/onnxruntime)
