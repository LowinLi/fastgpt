## 本例

+ 由于salesforce的codegen模型不兼容transformers，这里的例子是用fastgpt工具转codegen模型至onnx格式

## 处理流程

+ 1. 从CodeGen clone源代码
    ```shell
    git clone git@github.com:salesforce/CodeGen.git
    ```

+ 2. 下载模型
    ```shell
    wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-mono.tar.gz && tar -xvf checkpoints/codegen-350M-mono.tar.gz -C checkpoints/
    ```

+ 3. 安装fastgpt
    ```shell
    pip install fastgpt
    ```

+ 4. 开始转换
    ```shell
    python codegen_export_onnx.py
    ```

+ 5. 代码生成
    ```shell
    python codegen_onnx_inference.py
    ```

## docker部署

+ 1. 启动
```yaml
version: "2.3"
services:
  fastgpt-codegen:
    container_name: fastgpt-codegen
    image: lowinli98/fastgpt-codegen:v0.0.4
    expose:
      - 7104
    ports:
      - "7104:7104"
    environment:
      - PORT=7104
      - GUNICORN_WORKER=1
      - GUNICORN_THREADS=1
    restart: always
```

+ 2.测试

**codegen-350M-mono**
```
curl --location --request POST 'http://172.16.104.29:7104/generate_mono' \
--header 'Content-Type: application/json' \
--data-raw '{
    "inputs": "def calculdate_mean(x, y): \n",
    "parameters": {
        "do_sample": true
    }
}'
```
**codegen-350M-multi**
```
curl --location --request POST 'http://172.16.104.29:7104/generate_multi' \
--header 'Content-Type: application/json' \
--data-raw '{
    "inputs": "def calculdate_mean(x, y): \n",
    "parameters": {
        "do_sample": true
    }
}'
```
