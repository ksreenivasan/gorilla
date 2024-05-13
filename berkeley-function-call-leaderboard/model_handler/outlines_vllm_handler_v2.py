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
from tool_use.prompt import get_system_prompt
from tool_use.schema import tools_to_schema
from tool_use.tools import Tool

from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature


class OutlinesVllmHandler(BaseHandler):

    def __init__(
        self,
        model_name,
        temperature=0.7,
        top_p=1,
        max_tokens=150,
        seed=42,
        gen_mode="conditional",
        n_tool_calls=1,
        ) -> None:

        self.model_style = ModelStyle.Outlines
        self.n_tool_calls = n_tool_calls
        self.gen_mode = gen_mode
        super().__init__(model_name, temperature, top_p, max_tokens)

        # Initialize tool
        self.base_url = "http://localhost:8000/v1"
        self.api_key = "-"
        self.tool = Tool(self.base_url, self.api_key, self.model_name)

        self.idx = 0
        self.solutions = None

    def load_solutions(self, test_category):
        test_category = test_category.replace("executable_", "")
        solutions_path = f"./data/possible_answer/gorilla_openfunctions_v1_test_{test_category}.json"
        if not os.path.exists(solutions_path): # relevance tests
            solutions = [(0, 1)] * 400
            return solutions

        solutions = []
        with open(solutions_path, "r") as f:
            for line in f:
                solutions.append(json.loads(line))
        return solutions

    def inference(self, user_query, tools, test_category):

        # get number of tool calls from the solution
        if self.solutions is None:
            self.solutions = self.load_solutions(test_category)
        self.n_tool_calls = len(self.solutions[self.idx])
        self.idx += 1

        # Get schema for tool use
        try:
            tool_schema = tools_to_schema(tools)
        except Exception as e:
            result = f'[error.message(error="{str(e)}")]'
            print(f"An error occurred: {str(e)}")
            return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0, "n_tool_calls": self.n_tool_calls, "tool_calls": [], "messages": ""}

        # Prompt
        system_prompt = get_system_prompt(tool_schema)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            ]

        # Generate tool calls
        try:
            start = time.time()
            if self.gen_mode == "conditional":
                output_messages, tool_calls = self.tool.conditional(messages, tools, n_tool_calls=self.n_tool_calls)
            elif self.gen_mode == "structured":
                output_messages, tool_calls = self.tool.structured(messages, tools, n_tool_calls=self.n_tool_calls)
            elif self.gen_mode == "unstructured":
                output_messages, tool_calls = self.tool.unstructured(messages)

            result = bfcl_format(tool_calls)
        except Exception as e:
            result = f'[error.message(error="{str(e)}")]'
            print(f"An error occurred: {str(e)}")
            return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0,  "n_tool_calls": self.n_tool_calls, "tool_calls": [], "messages": ""}

        # Record info
        latency = time.time() - start
        metadata = {"input_tokens": 0, "output_tokens": 0, "latency": latency,  "n_tool_calls": self.n_tool_calls, "tool_calls": tool_calls, "messages": output_messages}
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

    def write(self, result, write_path):
        # if file_to_open[:-12] != "_result.json":
        #     file_to_open = file_to_open.replace(".json", "_result.json")

        # Make path if it does not exist
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        # Write path
        with open(write_path, "a+") as f:
            f.write(json.dumps(result) + "\n")

        # if not os.path.exists("./result"):
        #     os.mkdir("./result")
        # if not os.path.exists("./result/" + self.model_name.replace("/", "_")):
        #     os.mkdir("./result/" + self.model_name.replace("/", "_"))
        # with open("./result/" + self.model_name.replace("/", "_") + "/" + file_to_open, "a+") as f:
        #     f.write(json.dumps(result) + "\n")

    def load_result(self, test_category):
        # This method is used to load the result from the file.
        result_list = []
        with open(
            f"./result/{self.model_name.replace('/', '_')}/gorilla_openfunctions_v1_test_{test_category}_result.json"
        ) as f:
            for line in f:
                result_list.append(json.loads(line))
        return result_list

def bfcl_format(tool_calls):
    tool_strs = []
    for tool_call in tool_calls:
        args, name = tool_call["tool_arguments"], tool_call["tool_name"]
        args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in args.items()])
        tool_str = f'{name}({args_string})'
        tool_strs.append(tool_str)
    result = '[' + ', '.join(tool_strs) + ']'
    return result
