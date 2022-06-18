## 注意事项

步骤1、步骤2已经在代码中处理过了，不必再次执行

## 处理流程

+ 1. 从CodeGenclone源代码
    ```shell
    git clone git@github.com:salesforce/CodeGen.git
    ```
+ 2. 修改模型脚本支持onnx格式
    - onnx转换对torch.einsum计算时的不同数据类型支持不够
    + 参考: https://github.com/qhduan/CodeGen/blob/main/jaxformer/hf/codegen/modeling_codegen.py#L44
    + https://github.com/microsoft/onnxruntime/discussions/10121#discussioncomment-1987845

+ 3. 下载模型
    ```shell
    wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-multi.tar.gz && tar -xvf checkpoints/codegen-350M-multi.tar.gz -C checkpoints/
    ```

+ 4. 安装fastgpt
    ```shell
    pip install fastgpt
    ```

+ 5. 开始转换
    ```shell
    python codegen_export_onnx.py
    ```
