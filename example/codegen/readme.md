## 本例

+ 由于salesforce的codegen模型不兼容transformers，这里的例子是用fastgpt工具转codegen模型至onnx格式

## 处理流程

+ 1. 从CodeGenclone源代码
    ```shell
    git clone git@github.com:salesforce/CodeGen.git
    ```

+ 2. 下载模型
    ```shell
    wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-multi.tar.gz && tar -xvf checkpoints/codegen-350M-multi.tar.gz -C checkpoints/
    ```

+ 3. 安装fastgpt
    ```shell
    pip install fastgpt
    ```

+ 4. 开始转换
    ```shell
    python codegen_export_onnx.py
    ```
