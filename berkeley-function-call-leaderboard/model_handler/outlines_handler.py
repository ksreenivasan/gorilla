import json
import os
import re
import time
from functools import reduce
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


class OutlinesHandler(BaseHandler):

    def __init__(
        self,
        model_name,
        structured: bool = True,
        tokenizer_name = None,
        temperature=0.7,
        top_p=1,
        max_tokens=1000,
        n_tool_calls=1,
        seed=42) -> None:

        # self.tokenizer_name = tokenizer_name
        # if tokenizer_name is None:
        #     self.tokenizer_name = model_name
        self.tokenizer_name = model_name
        self.model_style = ModelStyle.Outlines

        self.llm = None
        self.tokenizer = None
        self.model = None

        self.structured = structured
        self.n_tool_calls = n_tool_calls

        self.seed = seed
        self.rng = torch.Generator(device="cuda")
        if self.seed:
            self.rng.manual_seed(self.seed)

        super().__init__(model_name, temperature, top_p, max_tokens)

    def _format_prompt_func(self, prompt, function):
        user_prompt = USER_PROMPT_FOR_CHAT_MODEL.format(
            user_prompt=prompt, functions=str(function)
            )
        system_prompt = SYSTEM_PROMPT_FOR_CHAT_MODEL

        #return f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT: "
        return user_prompt, system_prompt

    def inference(self, prompt, functions, test_category):

        # Only initialize model and tokenizer once
        if self.llm is None or self.tokenizer is None or self.model is None:
            self.llm = _init_llm(self.model_name)
            self.tokenizer = _init_tokenizer(self.tokenizer_name)
            self.model = models.Transformers(self.llm, self.tokenizer)

        # Cast to list
        if not isinstance(functions, list):
            functions = [functions]

        # Start timer
        start = time.time()

        # Initialize generator
        try:
            if self.structured:
                generator = get_regex_generator(functions, self.model, None, self.n_tool_calls, test_category)
            else:
                generator =  get_text_generator(self.model)
        except:
            result = '[error.message(error="Error occurred")]'
            return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0}

        # Format prompt
        apply_chat_template = False
        user_prompt = prompt
        prompt = format_prompt(user_prompt, self.tokenizer, functions, apply_chat_template)

        # Generate text with or without structure
        try:
            result = generator(prompt, rng=self.rng, max_tokens=self.max_tokens)
            if self.structured:
                result = format_result(result)
        except:
            result = '[error.message(error="Error occurred")]'
            return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0}

        # Record info
        latency = time.time() - start
        metadata = {"input_tokens": 0, "output_tokens": 0, "latency": latency}
        return result, metadata

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


def function_to_regex(function, test_category, whitespace_pattern, verbose):

    if isinstance(function, dict) and list(function.keys()) == ['name', 'description', 'parameters']:
        schema = bfcl_function_to_schema(function, test_category)
    elif callable(function):
        schema = json.dumps(get_schema_from_signature(function)) # get_schema_from_signature is from the outlines library
    else:
        raise TypeError
    if verbose >= 1:
        print(schema)

    schema_regex = build_regex_from_schema(schema, whitespace_pattern)
    function_regex = f'{{"function": "{function["name"]}", "arguments": {schema_regex}}}'

    if verbose >= 2:
        print(function_regex)

    return function_regex


def regex_or(pattern1, pattern2):
    return f"(?:{pattern1}|{pattern2})"


def repeat_pattern(pattern, num_repeats=None):
    """Repeat the regex pattern `pattern` `num_repeats` times.

    If `num_repeats` is `None`, allow the pattern to be repeated an unlimited number of times.
    If `num_repeats` is an integer, repeat the pattern exactly `num` times.
    If `num_repeats` is an iterable with length two, repeat the pattern anywhere between `num[0]` and `num[1]` times, inclusive.
    """
    if num_repeats is None:
        result = f"({pattern})*"
    elif isinstance(num_repeats, int):
        result = f"({pattern}){{{num_repeats}}}"
    elif isinstance(num_repeats, Union[list, tuple, set]) and len(num_repeats) == 2:
        return re.compile(f"({pattern}){{{num_repeats[0]},{num_repeats[1]}}}")
    return result


def all_functions_to_regex_str(functions, test_category, whitespace_pattern, n_tool_calls=None, verbose=0):
    """

    Specify the number of tools calls you can make.
    If `n_tool_calls` is `None`, allow the tool(s) to be repeated an unlimited number of times.
    If `n_tool_calls` is an integer, use exactly `n_tool_calls` tools.
    If `n_tool_calls` is an iterable with length two, repeat the pattern anywhere between `n_tool_calls[0]` and `n_tool_calls[1]` times, inclusive.
    """

    # Get a separate regex for each individual function
    function_regexes = [function_to_regex(function, test_category, whitespace_pattern, verbose) for function in functions]

    # Create a single regex that allows for any of the possible functions to be called
    function_regex = reduce(regex_or, function_regexes)

    # Allow multiple function calls or zero function calls
    function_regex = repeat_pattern(function_regex, n_tool_calls)

    if verbose >= 2:
        print(function_regex)

    return function_regex


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


def format_system_prompt(system_prompt, functions):

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



def format_prompt(user_prompt, tokenizer, functions, apply_chat_template, system_prompt=None):

    system_prompt = format_system_prompt(system_prompt, functions)

    # Setup chat template
    if apply_chat_template:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{system_prompt}\n{user_prompt}"

    return prompt


# def format_chat_template(tokenizer, user_prompt, system_prompt, tokenize=False, template=None):
#     """Tokenize the text with apply_tool_use_template, apply_chat_template, or with
#     no template.

#     If template is `None`, then tokenize first with tool use template, then chat template, and
#     then no template (in this order), if the tokenizer has these features. You can override
#     this by setting template to `tool_use` or `chat`.
#     """

#     if template == "tool_use" or (hasattr(tokenizer, "apply_tool_use_template") and template is None):
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#             ]
#         tools = None
#         inputs = tokenizer.apply_chat_template(
#             messages, tools=tools, tokenize=tokenize, add_generation_prompt=True,
#             return_tensors="pt", return_dict=True)
#     elif template == "chat" or (hasattr(tokenizer, "apply_chat_template") and template is None):
#         messages = [
#             # {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#             ]
#         inputs = tokenizer.apply_chat_template(
#             messages, tokenize=tokenize, add_generation_prompt=True, return_tensors="pt", return_dict=True)
#     else:
#         messages = [f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT: "]
#         inputs = messages
#         if tokenize:
#             inputs = tokenizer(messages, return_tensors="pt")

#     if tokenize:
#         device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#         inputs = inputs.to(device)
#     return inputs


def format_result(result):
    args, function_name = result["arguments"], result["function"]
    args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in args.items()])
    output_string = f'[{function_name}({args_string})]'
    return output_string
