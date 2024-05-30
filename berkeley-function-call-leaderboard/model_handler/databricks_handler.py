import json
import os
import re
import time

from model_handler.constant import (
    GORILLA_TO_OPENAPI,
    SYSTEM_PROMPT_FOR_CHAT_MODEL,
    USER_PROMPT_FOR_CHAT_MODEL,
)
from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import ast_parse, language_specific_pre_processing
from openai import OpenAI


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


class DatabricksHandler(BaseHandler):
    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        self.model_name = model_name
        self.model_style = ModelStyle.OpenAI
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # NOTE: To run the Databricks model, you need to provide your own Databricks API key and your own Azure endpoint URL.
        self.client = OpenAI(
            api_key=os.getenv("YOUR_DATABRICKS_API_KEY"),
            base_url=os.getenv("YOUR_DATABRICKS_AZURE_ENDPOINT_URL"),
        )

    def inference(self, prompt, functions, test_category):
        functions = language_specific_pre_processing(functions, test_category, False)
        if type(functions) is not list:
            functions = [functions]
        # message = [
        #     {"role": "system", "content": SYSTEM_PROMPT_FOR_CHAT_MODEL},
        #     {
        #         "role": "user",
        #         "content": "Questions:"
        #         + USER_PROMPT_FOR_CHAT_MODEL.format(
        #             user_prompt=prompt, functions=str(functions)
        #         ),
        #     },
        # ]


        # messages = [
        #     {"role": "system", "content": get_system_prompt(str(functions))},
        #     {"role": "user", "content": prompt},
        # ]

        # # function_input = {"type": "function", "function": functions[0]}
        # start_time = time.time()
        # response = self.client.chat.completions.create(
        #     messages=messages,
        #     model=self.model_name,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     top_p=self.top_p,
        # )


        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_CHAT_MODEL},
            {"role": "user", "content": USER_PROMPT_FOR_CHAT_MODEL.format(user_prompt=prompt, functions=str(functions))},
        ]
        # prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{SYSTEM_PROMPT_FOR_CHAT_MODEL}<|eot_id|><|start_header_id|>user<|end_header_id|>{USER_PROMPT_FOR_CHAT_MODEL.format(user_prompt=prompt, functions=str(functions))}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        start_time = time.time()
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        latency = time.time() - start_time
        result = response.choices[0].message.content
        metadata = {}
        metadata["input_tokens"] = response.usage.prompt_tokens
        metadata["output_tokens"] = response.usage.completion_tokens
        metadata["latency"] = latency
        return result, metadata

    def decode_ast(self, result, language="Python"):
        func = result
        func = func.replace("\n", "")  # remove new line characters
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"
        decoded_output = ast_parse(func, language)
        return decoded_output

    # def decode_ast(self, result, language="Python"):
    #     func = re.sub(r"'([^']*)'", r"\1", result)
    #     func = func.replace("\n    ", "")
    #     if not func.startswith("["):
    #         func = "[" + func
    #     if not func.endswith("]"):
    #         func = func + "]"
    #     if func.startswith("['"):
    #         func = func.replace("['", "[")
    #     try:
    #         decode_output = ast_parse(func, language)
    #     except:
    #         decode_output = ast_parse(result, language)
    #     return decode_output

    def decode_execute(self, result, language="Python"):
        func = re.sub(r"'([^']*)'", r"\1", result)
        func = func.replace("\n    ", "")
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"
        if func.startswith("['"):
            func = func.replace("['", "[")
        try:
            decode_output = ast_parse(func, language)
        except:
            decode_output = ast_parse(result, language)
        execution_list = []
        for function_call in decode_output:
            for key, value in function_call.items():
                execution_list.append(
                    f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                )
        return execution_list

    def write(self, result, write_path):
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        # Write path
        with open(write_path, "a+") as f:
            f.write(json.dumps(result) + "\n")
