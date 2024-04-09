import os
import json
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    convert_to_tool,
    ast_parse,
    augment_prompt_by_languge,
    language_specific_pre_processing,
    _function_calls_valid_format_and_invoke_extraction,
    _convert_value,
)
from model_handler.constant import (
    SYSTEM_PROMPT_FOR_CHAT_MODEL,
    USER_PROMPT_FOR_CHAT_MODEL,
    MODEL_ID_TO_NAME,
)

class AccelerateHandler(BaseHandler):

    def get_model_name(self, model_id):
        if model_id in MODEL_ID_TO_NAME:
            return MODEL_ID_TO_NAME[model_id]
        # Assume this is a HF model where the model name is after the last "/"
        return model_id.split("/")[-1]

    def __init__(self, model_id, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        self.model_id = model_id
        model_name = self.get_model_name(model_id)
        self.model_style = ModelStyle.Accelerate
        self.model = None
        self.tokenizer = None
        super().__init__(model_name, temperature, top_p, max_tokens)

    def _init_model(self):
        """From https://huggingface.co/docs/accelerate/usage_guides/big_modeling#complete-example"""
        checkpoint_dir = f"./model-checkpoints/{self.model_id}"
        snapshot_download(repo_id=self.model_id, local_dir=checkpoint_dir, token=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, use_auth_token=True)
        model = load_checkpoint_and_dispatch(model, checkpoint=checkpoint_dir, device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(
        #     self.model_id, device_map="auto", trust_remote_code=True, token=True
        #     )
        return model

    def _init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, token=True)
        return tokenizer

    def _format_prompt_func(self, prompt, function):
        user_prompt = USER_PROMPT_FOR_CHAT_MODEL.format(
            user_prompt=prompt, functions=str(function)
            )
        system_prompt = SYSTEM_PROMPT_FOR_CHAT_MODEL

        #return f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT: "
        return user_prompt, system_prompt

    def tokenizer_encode(self, user_prompt, system_prompt, tokenize=True, template=None):
        """Tokenize the text with apply_tool_use_template, apply_chat_template, or with
        no template.

        If template is `None`, then tokenize first with tool use template, then chat template, and
        then no template (in this order), if the tokenizer has these features. You can override
        this by setting template to `tool_use` or `chat`.
        """

        if template == "tool_use" or (hasattr(self.tokenizer, "apply_tool_use_template") and template is None):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
            tools = None
            inputs = self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=tokenize, add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
        elif template == "chat" or (hasattr(self.tokenizer, "apply_chat_template") and template is None):
            messages = [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        else:
            messages = [f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT: "]
            inputs = messages
            if tokenize:
                inputs = self.tokenizer(messages, return_tensors="pt")

        if tokenize:
            device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
            inputs = inputs.to(device)
        return inputs

    def inference(self, prompt, functions, test_category):

        # Only initialize model and tokenizer once
        if self.model is None and self.tokenizer is None:
            self.model = self._init_model()
            self.tokenizer = self._init_tokenizer()

        start = time.time()

        # Format the prompt with the functions the model has access to
        functions = language_specific_pre_processing(functions, test_category, False)
        prompt = augment_prompt_by_languge(prompt, test_category)
        user_prompt, system_prompt = self._format_prompt_func(prompt, functions)

        # Tokenize text
        input_tokens = self.tokenizer_encode(user_prompt, system_prompt)
        n_input_tokens = input_tokens["input_ids"].shape[1]

        # Generate text
        input_and_output_tokens = self.model.generate(**input_tokens, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=200)
        output_tokens = input_and_output_tokens[0][n_input_tokens:]
        n_output_tokens = len(output_tokens)

        # Decode text
        output_texts = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        # assert len(self.tokenizer(output_texts, add_special_tokens=False)["input_ids"]) == n_output_tokens

        # Record info
        latency = time.time() - start
        metadata = {
            "input_tokens": n_input_tokens,
            "output_tokens": n_output_tokens,
            "latency": latency
            }
        return output_texts, metadata

    def decode_execute(self, result):

        # Parse result
        func = result.replace('", "', ",").replace('["', "[").replace('"]', "]")
        if " " == func[0]:
            func = func[1:]
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"

        # Parse AST of func
        decode_output = ast_parse(func)

        # Create execution list
        execution_list = []
        for function_call in decode_output:
            for key, value in function_call.items():
                execution_list.append(
                    f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                )
        return execution_list
