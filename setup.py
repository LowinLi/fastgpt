import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="fastgpt",
    version="0.0.1",
    license="MIT License",
    author="LowinLi",
    author_email="lowinli@outlook.com",
    description="boost inference speed of GPT models in transformers by onnxruntime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LowinLi/fastgpt",
    project_urls={
        "Repo": "https://github.com/LowinLi/fastgpt",
        "Bug Tracker": "https://github.com/LowinLi/fastgpt/issues",
    },
    keywords=[
        "GPT",
        "ONNX",
        "onnxruntime",
        "NLP",
        "model hub" "transformer",
        "quantization",
        "text generation",
        "summarization",
        "translation",
        "q&a",
        "qg",
        "machine learning",
        "fast inference",
        "CausalLM",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "torch>=1.7.0,!=1.8.0",  # excludes torch v1.8.0
        "onnx",
        "onnxruntime==1.10.0",
        "numpy>=1.22.2",
        "transformers>4.6.1",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
