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

def format_result(self, function_name, result):
    # This method is used to format the result in a standard way.
    args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in result.items()])
    # Creating the output string with the function name and arguments
    output_string = f'[{function_name}({args_string})]'
    return output_string


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

        # Prompt
        regex_str, tool_schema = tool_to_regex(functions)
        system_prompt = get_system_prompt(tool_schema)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            ]

        # Maybe guided
        extra_body = {}
        if self.guided:
            extra_body=dict(guided_regex=regex_str, guided_decoding_backend="outlines")

        # Start timer
        start = time.time()

        # Generate text
        completion = client.chat.completions.create(
            model="databricks/dbrx-instruct",
            messages=messages,
            extra_body=extra_body,
            )

        # Parse output
        raw_text = completion.choices[0].message.content
        tool = json.loads(raw_text)
        result = format_result(tool["tool_name"], tool["tool_arguments"])

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


def get_system_prompt(
    tool_schema,
    tool_list_start="<tool>",
    tool_list_end="</tools>",
    tool_call_start="<tool_call>",
    tool_call_end="</tool_call>",
    tool_response_start="<tool_response>",
    tool_response_end="</tool_response>"
    ):

    system_prompt = """You are a function calling AI model. Your job is to answer the user's questions and you may call one or more functions to do this.


    Please use your own judgment as to whether or not you should call a function. In particular, you must follow these guiding principles:
    1. You may call one or more functions to assist with the user query.
    2. You do not need to call a function. If none of the functions can be used to answer the user's question, please do not make the function call.
    3. Don't make assumptions about what values to plug into functions. If you are missing the parameters to make a function call, please ask the user for the parameters.
    4. You may assume the user has implemented the function themselves.
    5. You may assume the user will call the function on their own. You should NOT ask the user to call the function and let you know the result; they will do this on their own.


    You can only call functions according the following formatting rules:
    Rule 1: All the functions you have access to are contained within {tool_list_start}{tool_list_end} XML tags. You cannot use any functions that are not listed between these tags.

    Rule 2: For each function call return a json object (using quotes) with function name and arguments within {tool_call_start}\n{{ }}\n{tool_call_end} XML tags as follows:
    * With arguments:
    {tool_call_start}\n{{"name": "function_name", "arguments": {{"argument_1_name": "value", "argument_2_name": "value"}} }}\n{tool_call_end}
    * Without arguments:
    {tool_call_start}\n{{ "name": "function_name", "arguments": {{}} }}\n{tool_call_end}
    In between {tool_call_start} and{tool_call_end} tags, you MUST respond in a valid JSON schema.
    In between the {tool_call_start} and {tool_call_end} tags you MUST only write in json; no other text is allowed.

    Rule 3: If user decides to run the function, they will output the result of the function call between the {tool_response_start} and {tool_response_start} tags. If it answers the user's question, you should incorporate the output of the function in your answer.


    Here are the tools available to you:
    {tool_list_start}\n{tool_schema}\n{tool_list_end}

    Remember, don't make assumptions about what values to plug into functions. If you are missing the parameters to make a function call, please ask the user for the parameters. Do not be afraid to ask.
    """

    return system_prompt.format(
        tool_list_start=tool_list_start,
        tool_list_end=tool_list_end,
        tool_call_start=tool_call_start,
        tool_call_end=tool_call_end,
        tool_response_start=tool_response_start,
        tool_response_end=tool_response_end,
        tool_schema=tool_schema,
        )


def format_result(result):
    args, function_name = result["arguments"], result["function"]
    args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in args.items()])
    output_string = f'[{function_name}({args_string})]'
    return output_string
