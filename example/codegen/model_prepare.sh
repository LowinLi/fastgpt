# prepare repo
git clone https://github.com/salesforce/CodeGen.git

# prepare model
mkdir -p checkpoints/
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-mono.tar.gz && tar -xvf checkpoints/codegen-350M-mono.tar.gz -C checkpoints/ && rm checkpoints/codegen-350M-mono.tar.gz
wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-multi.tar.gz && tar -xvf checkpoints/codegen-350M-multi.tar.gz -C checkpoints/ && rm checkpoints/codegen-350M-multi.tar.gz
python3 codegen_export_onnx.py --model_path checkpoints/codegen-350M-mono
python3 codegen_export_onnx.py --model_path checkpoints/codegen-350M-multi
rm checkpoints/codegen-350M-mono/pytorch_model.bin
rm checkpoints/codegen-350M-mono/onnx/model.onnx
rm checkpoints/codegen-350M-multi/pytorch_model.bin
rm checkpoints/codegen-350M-multi/onnx/model.onnx