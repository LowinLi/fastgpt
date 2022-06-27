import setuptools
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))
version = (
    subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
    .decode("utf-8")
    .strip()
)
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="fastgpt",
    version=version,
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
        "torch>=1.10.0",
        "onnx",
        "onnxruntime==1.10.0",
        "numpy>=1.22.2",
        "transformers>=4.19",
        "six==1.16.0",
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
