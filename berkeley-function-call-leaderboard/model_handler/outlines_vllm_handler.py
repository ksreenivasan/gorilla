import json
import os
import re
import time
from functools import reduce
from textwrap import dedent
from typing import Union

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from model_handler.constant import (
    GORILLA_TO_OPENAPI,
    SYSTEM_PROMPT_FOR_CHAT_MODEL,
    USER_PROMPT_FOR_CHAT_MODEL,
)
from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    _cast_to_openai_type,
    _convert_value,
    _function_calls_valid_format_and_invoke_extraction,
    ast_parse,
    augment_prompt_by_languge,
    convert_to_tool,
    language_specific_pre_processing,
)
from outlines import generate, models
from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


class OutlinesVllmHandler(BaseHandler):

    def __init__(
        self,
        model_name,
        temperature=0.7,
        top_p=1,
        max_tokens=150,
        guided: bool = True,
        n_tool_calls=1,
        seed=42) -> None:


        # self.base_url = os.getenv("BASE_URL")
        # self.api_key = os.getenv("API_KEY")
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
        self.model_style = ModelStyle.Outlines

        self.guided = guided
        self.n_tool_calls = n_tool_calls

        self.seed = seed
        self.rng = torch.Generator(device="cuda")
        if self.seed:
            self.rng.manual_seed(self.seed)

        super().__init__(model_name, temperature, top_p, max_tokens)

    def inference(self, prompt, functions, test_category):

        # Cast to list
        if not isinstance(functions, list):
            functions = [functions]

        # Start timer
        start = time.time()

        extra_body = {}
        if self.guided:
            regex_str, tool_schema = tool_to_regex(tools)

            messages =
            extra_body=dict(guided_regex=regex_str, guided_decoding_backend="outlines")

        # Initialize generator
        completion = client.chat.completions.create(
            model="databricks/dbrx-instruct",
            messages=messages,
            extra_body=extra_body,
            )

raw_text = completion.choices[0].message.content


        # Record info
        latency = time.time() - start
        metadata = {"input_tokens": 0, "output_tokens": 0, "latency": latency}
        return result, metadata

    def decode_ast(self, result, language="Python"):
        decoded_output = ast_parse(result, language)
        return decoded_output

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

    def write(self, result, file_to_open):
        # if file_to_open[:-12] != "_result.json":
        #     file_to_open = file_to_open.replace(".json", "_result.json")

        if not os.path.exists("./result"):
            os.mkdir("./result")
        if not os.path.exists("./result/" + self.model_name.replace("/", "_")):
            os.mkdir("./result/" + self.model_name.replace("/", "_"))
        with open("./result/" + self.model_name.replace("/", "_") + "/" + file_to_open, "a+") as f:
            f.write(json.dumps(result) + "\n")

    def load_result(self, test_category):
        # This method is used to load the result from the file.
        result_list = []
        with open(
            f"./result/{self.model_name.replace('/', '_')}/gorilla_openfunctions_v1_test_{test_category}_result.json"
        ) as f:
            for line in f:
                result_list.append(json.loads(line))
        return result_list


#####################################################################
# Function to Regex
#####################################################################

def bfcl_function_to_schema(function, test_category):
    properties = _cast_to_openai_type(function["parameters"]["properties"], GORILLA_TO_OPENAPI, test_category)
    schema = json.dumps({
        "title": function["name"],
        "type": "object",
        "description": function["description"],
        "properties": properties,
        "required": function["parameters"]["required"],
        })
    return schema


def regex_or(pattern1, pattern2):
    return f"(?:{pattern1}|{pattern2})"


def sometime_guide(regex_pattern, start_guided_pattern="<tool_call>", end_guided_pattern="</tool_call>"):
    """
    Only do guided generation sometimes, in between start_word and end_word.
    """
    return f".*?(?={start_guided_pattern}){start_guided_pattern}({regex_pattern}).*?(?={end_guided_pattern}){end_guided_pattern}.*"


def is_bfcl(tool):
    return isinstance(tool, dict) and list(tool.keys()) == ['name', 'description', 'parameters']


def tool_to_regex(tool, whitespace_pattern=None, test_category=None):

    if isinstance(tool, list):
        values = [tool_to_regex(_tool, whitespace_pattern=whitespace_pattern, test_category=test_category) for _tool in tool]
        regex_strs = [v[0] for v in values]
        regex_str = reduce(regex_or, regex_strs)
        schema_strs = [v[1] for v in values]
        schema_str = "\n".join(schema_strs)
    elif is_bfcl(tool):
        schema_str = bfcl_function_to_schema(tool, test_category).strip()
        schema_regex = build_regex_from_schema(schema_str, whitespace_pattern)
        regex_str = f'{{"tool_name": "{tool["name"]}", "tool_arguments": {schema_regex}}}'
    elif isinstance(tool, type(BaseModel)):
        schema_json = tool.model_json_schema()
        schema_str = json.dumps(schema_json).strip()
        schema_regex = build_regex_from_schema(schema_str, whitespace_pattern)
        regex_str = f'{{"tool_name": "{schema_json["title"]}", "tool_arguments": {schema_regex}}}'
    elif callable(tool):
        schema_json = get_schema_from_signature(tool)
        schema_str = json.dumps(schema_json).strip()
        schema_regex = build_regex_from_schema(schema_str, whitespace_pattern)
        regex_str = f'{{"tool_name": "{tool.__name__}", "tool_arguments": {schema_regex}}}'
    elif isinstance(tool, str):
        schema_str = tool.replace("\n", " ").replace("  ", " ").strip()
        schema_regex = build_regex_from_schema(schema_str, whitespace_pattern)
        regex_str = f'{{"tool_name": "{json.loads(schema_str)["title"]}", "tool_arguments": {schema_regex}}}'

    return regex_str, schema_str


#####################################################################
# Initialize model + tokenizer
#####################################################################


def _init_llm(model_name):
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token="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",
        )
    return llm


def _init_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        token="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",
        )
    return tokenizer


#####################################################################
# Generator
#####################################################################


def get_regex_generator(functions, model, whitespace_pattern, n_tool_calls, test_category, verbose=0):
    functions_regex_str = all_functions_to_regex_str(
        functions,
        test_category,
        whitespace_pattern,
        n_tool_calls=n_tool_calls,
        verbose=verbose,
        )

    generator = generate.regex(model, functions_regex_str)
    # generator.format_sequence = lambda x: x
    generator.format_sequence = lambda x: json.loads(x) # to work with json; from https://github.com/outlines-dev/outlines/blob/078f8223b6d8970ca6cc12d6c17659868e993691/outlines/generate/json.py#L60

    return generator

def get_text_generator(model):
    return generate.text(model)


#####################################################################
# Prompt
#####################################################################


def get_system_prompt(system_prompt, functions):

    # Format functions as string
    functions_str = "\n".join([str(function) for function in functions])

    # Setup system prompt
    if system_prompt is None:
        system_prompt_tuple = (
            "You are a helpful assistant and an expert in function calling.",
            "You have access to several functions which are represented in json schemas.",
            "Here are the functions:\n{functions}\n",
            "If you are requested to use a function, you ALWAYS output functions in a valid json schema."
            "If there are no relevant functions, please ask for more information before making the function call."
            )
        system_prompt = ' '.join(system_prompt_tuple)
        system_prompt = system_prompt.format(functions=functions_str)
    elif system_prompt is False:
        system_prompt = ""
    else: # system_prompt is a string with `{functions}` in it
        system_prompt = system_prompt.format(functions=functions_str)
    return system_prompt

def get_gemma_prompt(user_prompt, tokenizer, functions, apply_chat_template, system_prompt=None):

    gemma_prompt_template = (
    "<bos><start_of_turn>user\n",
    "You are a helpful assistant and an expert in function calling.",
    "A user is gonna ask you a question, you need to extract the arguments to be passed to the function that can answer the question.",
    "You have access to several functions which are represented in json schemas. Here are the functions:\n{functions}\n",
    "The user's question below:\n{question}\n",
    "If you are requested to use a function, you ALWAYS output functions in a valid json schema. You must answer the user's question by replying VALID JSON.",
    "<end_of_turn>\n",
    "<start_of_turn>model\n",
    )

    functions_str = "\n".join([str(function) for function in functions])
    gemma_prompt_template = ' '.join(gemma_prompt_template)
    return gemma_prompt_template.format(functions=functions_str, question=user_prompt)

    system_prompt = format_system_prompt(system_prompt, functions)

    # Setup chat template
    if apply_chat_template:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{system_prompt}\n{user_prompt}"

    return prompt


def use_chat_template(tokenizer, user_prompt, system_prompt, template=None):
    """Format the text with apply_tool_use_template, apply_chat_template, or with
    no template.

    If template is `None`, then tokenize first with tool use template, then chat template, and
    then no template (in this order), if the tokenizer has these features. You can override
    this by setting template to `tool_use` or `chat`.
    """

    if template == "tool_use" or (hasattr(tokenizer, "apply_tool_use_template") and template is None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            ]
        tools = None
        inputs = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    elif template == "chat" or (hasattr(tokenizer, "apply_chat_template") and template is None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages = f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT: "
        inputs = messages

    return inputs


def format_result(result):
    args, function_name = result["arguments"], result["function"]
    args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in args.items()])
    output_string = f'[{function_name}({args_string})]'
    return output_string
