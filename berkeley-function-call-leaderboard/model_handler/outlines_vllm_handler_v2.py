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
from tool_use.tool import Tool

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
        n_tool_calls_mode="solution",
        ) -> None:

        self.model_style = ModelStyle.Outlines
        self._n_tool_calls = n_tool_calls
        self.n_tool_calls = None
        self.gen_mode = gen_mode
        super().__init__(model_name, temperature, top_p, max_tokens)

        # Initialize tool
        self.base_url = "http://localhost:8000/v1"
        self.api_key = "-"
        self.tool = Tool(self.base_url, self.api_key, self.model_name)

    def load_solution(self, user_query, test_category):
        test_category = test_category.replace("executable_", "")
        solutions_path = f"./data/possible_answer/gorilla_openfunctions_v1_test_{test_category}.json"
        questions_path = f"./data/gorilla_openfunctions_v1_test_{test_category}.json"

        # Find the index where the question matches the user query
        question_idx = 0
        with open(questions_path, "r") as f:
            for line in f:
                question = json.loads(line)["question"]
                if question == user_query:
                    break
                question_idx += 1

        # Get solution located at question_idx
        solution_idx = 0
        solution = {}
        with open(solutions_path, "r") as f:
            for line in f:
                if solution_idx == question_idx:
                    solution = json.loads(line)
                    break
                solution_idx += 1

        return solution

    def inference(self, user_query, tools, test_category):

        # get n_tool_calls
        if self._n_tool_calls == "solution":
            self.solution = self.load_solution(user_query, test_category)
            self.n_tool_calls = len(self.solution)
        else:
            self.n_tool_calls = self._n_tool_calls

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
            output_messages, tool_calls = self.tool(messages, tools, gen_mode=self.gen_mode, n_tool_calls=self.n_tool_calls)
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
        os.makedirs(os.path.dirname(write_path), exist_ok=True)

        # Write path
        with open(write_path, "a+") as f:
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

def bfcl_format(tool_calls):
    tool_strs = []
    for tool_call in tool_calls:
        args, name = tool_call["tool_arguments"], tool_call["tool_name"]
        args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in args.items()])
        tool_str = f'{name}({args_string})'
        tool_strs.append(tool_str)
    result = '[' + ', '.join(tool_strs) + ']'
    return result
