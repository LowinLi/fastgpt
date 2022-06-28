from flask import Flask, request, Response
import copy
import simplejson
from logger import logger
from codegen_onnx_inference import get_codegen_model_tokenizer

mono_model, tokenizer = get_codegen_model_tokenizer("codegen-350M-mono")
multi_model, tokenizer = get_codegen_model_tokenizer("codegen-350M-multi")

default_parameters = {
    "temp": 0.2,
    "top_p": 0.95,
    "top_k": 5,
    "max_length_sample": 128,
    "max_length": 2048,
}

app = Flask(__name__)


@app.route("/generate_mono", methods=["POST"])
def generate_mono():

    body = request.json
    inputs = body.get("inputs", "")
    parameters = copy.deepcopy(default_parameters)
    restful_parameters = body.get("parameters", {})
    parameters.update(restful_parameters)
    input_ids = tokenizer(
        inputs,
        truncation=True,
        padding=True,
        max_length=parameters["max_length"],
        return_tensors="pt",
    ).input_ids
    tokens = mono_model.generate(
        input_ids,
        do_sample=False,
        num_return_sequences=1,
        temperature=parameters["temp"],
        max_length=input_ids.shape[1] + parameters["max_length_sample"],
        top_p=parameters["top_p"],
        top_k=parameters["top_k"],
        use_cache=True,
        early_stopping=True
    )
    gen_text = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True) # 只要新生成的
    logger.info(f"###\ninput:{inputs}\noutput:{gen_text}")
    data = simplejson.dumps(
        [{"generated_text": gen_text}], indent=4, ensure_ascii=False, ignore_nan=True
    )
    return Response(data, mimetype="application/json")


@app.route("/generate_multi", methods=["POST"])
def generate_multi():

    body = request.json
    inputs = body.get("inputs", "")
    parameters = copy.deepcopy(default_parameters)
    restful_parameters = body.get("parameters", {})
    parameters.update(restful_parameters)
    input_ids = tokenizer(
        inputs,
        truncation=True,
        padding=True,
        max_length=parameters["max_length"],
        return_tensors="pt",
    ).input_ids
    tokens = multi_model.generate(
        input_ids,
        do_sample=False,
        num_return_sequences=1,
        temperature=parameters["temp"],
        max_length=input_ids.shape[1] + parameters["max_length_sample"],
        top_p=parameters["top_p"],
        use_cache=True,
    )
    gen_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    logger.info(f"###\ninput:{inputs}\noutput:{gen_text}")
    data = simplejson.dumps(
        [{"generated_text": gen_text}], indent=4, ensure_ascii=False, ignore_nan=True
    )
    return Response(data, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
