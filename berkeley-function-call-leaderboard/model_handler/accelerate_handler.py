import os
import json
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    convert_to_tool,
    ast_parse,
    augment_prompt_by_languge,
    language_specific_pre_processing,
    construct_tool_use_system_prompt,
    _function_calls_valid_format_and_invoke_extraction,
    _convert_value,
)
from model_handler.constant import (
    SYSTEM_PROMPT_FOR_CHAT_MODEL,
    USER_PROMPT_FOR_CHAT_MODEL,
)

class AccelerateHandler(BaseHandler):
    def __init__(self, model_name, checkpoint_dir=".checkpoint/", temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_style = ModelStyle.Accelerate
        self.checkpoint_dir = checkpoint_dir
        self.model = self._init_model()
        self.tokenizer = self._init_tokenizer()

    def _init_model(self):
        """From https://huggingface.co/docs/accelerate/usage_guides/big_modeling#complete-example"""
        weights_location = hf_hub_download(self.model_name, "pytorch_model.bin", token=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, use_auth_token=True)
        model = load_checkpoint_and_dispatch(model, checkpoint=weights_location, device_map="auto")
        return model

    def _init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_auth_token=True,
                truncation_side="left",
                padding_side="right",
            )
        return tokenizer

    def _format_prompt_func(self, prompt, function):
        SYSTEM_PROMPT = """You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed-"""
        functions = ""
        if isinstance(function, list):
            for idx, func in enumerate(function):
                functions += "\n" + str(func)
        else:
            functions += "\n" + str(function)
        return f"SYSTEM: {SYSTEM_PROMPT}\n{functions}\nUSER: {prompt}\nASSISTANT: "


    def inference(self, prompt, functions, test_category):
        start = time.time()

        # Format the prompt with the functions the model has access to
        functions = language_specific_pre_processing(functions, test_category, False)
        prompt = augment_prompt_by_languge(prompt, test_category)
        prompt = self._format_prompt_func(prompt, functions)

        # Tokenize, generate, decode
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        result = self.tokenizer.decode(outputs[0])

        # Record info
        latency = time.time() - start
        metadata = {
            "input_tokens": prompt,
            "output_tokens": result[len(prompt):],
            "latency": latency
            }
        return result, metadata

    def write(self, result, file_to_open):
        model_name = self.model_name
        model_name_path =  model_name.replace("/", "_")
        if not os.path.exists("./result"):
            os.mkdir("./result")
        out_dir = os.path.join("./result", model_name_path)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, file_to_open), "a+") as f:
            f.write(json.dumps(result) + "\n")
