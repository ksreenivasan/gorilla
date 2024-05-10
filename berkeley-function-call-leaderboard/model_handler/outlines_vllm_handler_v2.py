import json
import os
import re
import time
from functools import reduce
from typing import Union

import torch
from model_handler.constant import GORILLA_TO_OPENAPI
from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import _cast_to_openai_type, ast_parse
from openai import OpenAI
from pydantic import BaseModel
from tool_use.schema import tools_to_schema
from tool_use.tools import Tool

from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature


class OutlinesVllmHandler(BaseHandler):

    def __init__(
        self,
        model_name,
        temperature=0.7,
        top_p=1,
        max_tool_calls=5,
        max_tokens=150,
        guided: bool = True,
        seed=42) -> None:

        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
        self.model_style = ModelStyle.Outlines

        self.guided = guided
        self.seed = seed
        self.rng = torch.Generator(device="cuda")
        if self.seed:
            self.rng.manual_seed(self.seed)

        self.max_tool_calls = max_tool_calls

        super().__init__(model_name, temperature, top_p, max_tokens)

    def inference(self, prompt, functions, test_category):

        # Cast to list
        if not isinstance(functions, list):
            functions = [functions]

        # Get regex for tool use
        try:
            regex_str, tool_schema = tool_to_regex(functions)
        except Exception as e:
            result = f'[error.message(error="{str(e)}")]'
            print(f"An error occurred: {str(e)}")
            return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0, "messages": "", "tool_calls": []}

        # Prompt
        system_prompt = get_system_prompt(tool_schema)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            ]

        # Generate tool calls
        try:
            start = time.time()
            messages, tool_calls = get_tool_calls(self.client, self.model_name, messages, regex_str, max_tool_calls=self.max_tool_calls)
            result = bfcl_format(tool_calls)
        except Exception as e:
            result = f'[error.message(error="{str(e)}")]'
            print(f"An error occurred: {str(e)}")
            return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0, "messages": "", "tool_calls": []}

        # Record info
        latency = time.time() - start
        metadata = {"input_tokens": 0, "output_tokens": 0, "latency": latency, "messages": messages, "tool_calls": tool_calls}
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
# Tool to Regex
#####################################################################


GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}


def _cast_to_openai_type(properties, mapping, test_category):
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]
            if mapping == GORILLA_TO_OPENAPI and var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] += " This is a float type value."
            if var_type in mapping:
                properties[key]["type"] = mapping[var_type]
            else:
                properties[key]["type"] = "string"

        # Currently support:
        # - list of any
        # - list of list of any
        # - list of dict
        # - list of list of dict
        # - dict of any

        if properties[key]["type"] == "array" or properties[key]["type"] == "object":
            if "properties" in properties[key]:
                properties[key]["properties"] = _cast_to_openai_type(
                    properties[key]["properties"], mapping, test_category
                )
            elif "items" in properties[key]:
                properties[key]["items"]["type"] = mapping[
                    properties[key]["items"]["type"]
                ]
                if (
                    properties[key]["items"]["type"] == "array"
                    and "items" in properties[key]["items"]
                ):
                    properties[key]["items"]["items"]["type"] = mapping[
                        properties[key]["items"]["items"]["type"]
                    ]
                elif (
                    properties[key]["items"]["type"] == "object"
                    and "properties" in properties[key]["items"]
                ):
                    properties[key]["items"]["properties"] = _cast_to_openai_type(
                        properties[key]["items"]["properties"], mapping, test_category
                    )
    return properties


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
    Only do guided generation sometimes, i.e. only force us to output according to the regex pattern in between start_word and end_word.
    """
    return f".*?(?={start_guided_pattern}){start_guided_pattern}({regex_pattern}).*?(?={end_guided_pattern}){end_guided_pattern}.*"


def is_bfcl(tool):
    return isinstance(tool, dict) and list(tool.keys()) == ['name', 'description', 'parameters']


def repeat_regex_pattern(pattern, num_repeats, sep="\\n"):
    """Repeat the regex pattern `pattern` `num_repeats` times.

    If `num_repeats` is `None`, allow the pattern to be repeated an unlimited number of times.
    If `num_repeats` is an integer, repeat the pattern exactly `num` times.
    If `num_repeats` is an iterable with length two, repeat the pattern anywhere between `num[0]` and `num[1]` times, inclusive.
    """

    if num_repeats is None:
        min_repetitions = 0
        max_repetitions = None
    elif isinstance(num_repeats, int):
        min_repetitions = max_repetitions = num_repeats
    elif isinstance(num_repeats, Union[list, tuple, set]) and len(num_repeats) == 2:
        min_repetitions = num_repeats[0]
        max_repetitions = num_repeats[1]

    if max_repetitions is None:
        regex_str = f'({pattern}{sep}){{{min_repetitions},}}'
    else:
        regex_str = f'({pattern}{sep}){{{min_repetitions},{max_repetitions}}}'

    return regex_str


def tool_to_regex(
    tool,
    n_tool_calls=1,
    tool_call_start="<tool_call>",
    tool_call_end="</tool_call>",
    sometimes=False,
    whitespace_pattern=None,
    test_category=None,
    ):

    if isinstance(tool, list):
        values = [
            tool_to_regex(_tool, n_tool_calls=n_tool_calls, tool_call_start=tool_call_start, tool_call_end=tool_call_end, sometimes=sometimes, whitespace_pattern=whitespace_pattern, test_category=test_category,)
            for _tool in tool
            ]
        regex_strs, schema_strs = [v[0] for v in values], [v[1] for v in values]
        regex_str = reduce(regex_or, regex_strs)
        schema_str = "\n".join(schema_strs)
    elif is_bfcl(tool): # only works for BFCL
        schema_json = {
            "title": tool["name"],
            "type": "object",
            "description": tool["description"],
            "properties": tool["parameters"]["properties"],
            "required": tool["parameters"]["required"],
            }
        schema_str = json.dumps(schema_json)
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
        schema_str = re.sub(r'\s+', ' ', tool).strip()
        schema_regex = build_regex_from_schema(schema_str, whitespace_pattern)
        regex_str = f'{{"tool_name": "{json.loads(schema_str)["title"]}", "tool_arguments": {schema_regex}}}'

    # if sometimes:
    #     regex_str = sometime_guide(regex_str)
    if not isinstance(tool, list):
        # regex_str = f"{tool_call_start}{regex_str}{tool_call_end}"
        regex_str = f"{regex_str}{tool_call_end}"

    # if not isinstance(tool, list):
    #     regex_str = repeat_regex_pattern(regex_str, n_tool_calls)

    return regex_str, schema_str


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
    1. You may call one or more functions to assist with the user query. You should call multiple functions when the user asks you to.
    2. You do not need to call a function. If none of the functions can be used to answer the user's question, please do not make the function call.
    3. Don't make assumptions about what values to plug into functions. If you are missing the parameters to make a function call, please ask the user for the parameters.
    4. You may assume the user has implemented the function themselves.
    5. You may assume the user will call the function on their own. You should NOT ask the user to call the function and let you know the result; they will do this on their own.
    6. Never call a tool twice with the same exact arguments. Do not repeat your tool calls!


    You can only call functions according the following formatting rules:
    Rule 1: All the functions you have access to are contained within {tool_list_start}{tool_list_end} XML tags. You cannot use any functions that are not listed between these tags.

    Rule 2: For each function call return a json object (using quotes) with function name and arguments within {tool_call_start}\n{{ }}\n{tool_call_end} XML tags as follows:
    * With arguments:
    {tool_call_start}\n{{"tool_name": "function_name", "tool_arguments": {{"argument_1_name": "value", "argument_2_name": "value"}} }}\n{tool_call_end}
    * Without arguments:
    {tool_call_start}\n{{ "tool_name": "function_name", "tool_arguments": {{}} }}\n{tool_call_end}
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


#####################################################################
# Generator
#####################################################################

def generate_structured(client, model_name, messages, regex_str, stop_token=None, max_tokens=4096):

    completion = client.chat.completions.create(
      model=model_name,
      max_tokens=max_tokens,
      messages=messages,
      stop=stop_token,
      temperature=0,
      extra_body=dict(guided_regex=regex_str, guided_decoding_backend="outlines"),
      )
    raw_text = completion.choices[0].message.content
    finish_reason = completion.choices[0].stop_reason
    return raw_text, finish_reason


def generate_unstructured(client, model_name, messages, stop_token=None, max_tokens=4096):

    completion = client.chat.completions.create(
      model=model_name,
      max_tokens=max_tokens,
      messages=messages,
      stop=stop_token,
      temperature=0,
      extra_body={},
      )
    raw_text = completion.choices[0].message.content
    finish_reason = completion.choices[0].stop_reason
    return raw_text, finish_reason


def get_tool_calls(client, model_name, messages, regex_str, tool_call_start="<tool_call>", tool_call_end="</tool_call>", max_tool_calls=5, verbose=0):

    n_tool_calls = 0
    tool_calls = []

    text, finish_reason = generate_unstructured(client, model_name, messages, stop_token=tool_call_start)
    text += tool_call_start
    messages.append({"role": "assistant", "content": text})
    if verbose: print("-"*70, "\n", "(Finish:", finish_reason, ")\n", text)

    while n_tool_calls < max_tool_calls and finish_reason == tool_call_start:

        text, finish_reason = generate_structured(client, model_name, messages, stop_token=tool_call_end, regex_str=regex_str)
        tool_calls.append(json.loads(text))
        text += tool_call_end
        messages.append({"role": "assistant", "content": text})
        if verbose: print("-"*70, "\n", "(Finish:", finish_reason, ")\n", text)

        n_tool_calls += 1

        text, finish_reason = generate_unstructured(client, model_name, messages, stop_token=tool_call_start)
        text += tool_call_start
        messages.append({"role": "assistant", "content": text})
        if verbose: print("-"*70, "\n", "(Finish:", finish_reason, ")\n", text)

    return messages, tool_calls


#####################################################################
# Parse
#####################################################################

def bfcl_format(tool_calls):
    tool_strs = []
    for tool_call in tool_calls:
        args, name = tool_call["tool_arguments"], tool_call["tool_name"]
        args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in args.items()])
        tool_str = f'{name}({args_string})'
        tool_strs.append(tool_str)
    result = '[' + ', '.join(tool_strs) + ']'
    return result
