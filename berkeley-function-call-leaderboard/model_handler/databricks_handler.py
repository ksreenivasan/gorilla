from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    language_specific_pre_processing,
    ast_parse,
    convert_to_tool,
    augment_prompt_by_languge,
    convert_to_function_call,
)
from model_handler.constant import (
    SYSTEM_PROMPT_FOR_CHAT_MODEL,
    USER_PROMPT_FOR_CHAT_MODEL,
    GORILLA_TO_OPENAPI,
    USER_PROMPT_FOR_CHAT_MODEL_FC,
    SYSTEM_PROMPT_FOR_CHAT_MODEL_FC
)
import time
from openai import OpenAI
import re
import os
import json

from tool_use.tool import Tool


class DatabricksHandler(BaseHandler):
    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        self.model_name = model_name
        self.model_style = ModelStyle.OpenAI
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # # NOTE: To run the Databricks model, you need to provide your own Databricks API key and your own Azure endpoint URL.
        # self.client = OpenAI(
        #     api_key=os.environ["DATABRICKS_API_KEY"],
        #     base_url=os.environ["DATABRICKS_ENDPOINT_URL"],
        # )
        self.tool = Tool("http://localhost:8080/v1", "-", "databricks/dbrx-instruct")

    def inference(self, prompt, functions, test_category):
        # TODO: do this more elegantly
        API_FAILURE_MESSAGE = None # hacky way to get the error message out of the try block
        if "FC" not in self.model_name:
            functions = language_specific_pre_processing(functions, test_category, False)
            if type(functions) is not list:
                functions = [functions]
            message = [
                {"role": "system", "content": SYSTEM_PROMPT_FOR_CHAT_MODEL},
                {
                    "role": "user",
                    "content": "Questions:"
                    + USER_PROMPT_FOR_CHAT_MODEL.format(
                        user_prompt=prompt, functions=str(functions)
                    ),
                },
            ]
            start_time = time.time()
            latency = time.time() - start_time
            # result = response.choices[0].message.content
        else:
            # TODO: see if this is necessary. they do it for openai models
            # prompt = augment_prompt_by_languge(prompt, test_category)
            functions = language_specific_pre_processing(functions, test_category, True)
            if type(functions) is not list:
                functions = [functions]
            message = [{"role": "user", "content": prompt}]

            # NOTE: since we're using the deprecated function_call api, we don't
            # need to convert it to "tools". But this method also modifies the functions
            # list to make it adhere to the json schema. So I guess I need to run it anyway lol.
            # TODO: Fix this.
            oai_tool = convert_to_tool(
                functions, GORILLA_TO_OPENAPI, self.model_style, test_category, True
            )

            start_time = time.time()
            if len(functions) > 0:
                try:
                    print(message, oai_tool)
                    tool_use_tool = [ i.get("function") for i in oai_tool]
                    output_messages, tool_calls = self.tool(message, tool_use_tool, gen_mode="meta_tool", n_tool_calls=1)
                    print(f"\nTool calls: {tool_calls}\n")
                except Exception as e:
                    print(f"\nError while trying to do FC: {e}\n")
                    print(f"Messages={message}")
                    print(f"Functions={functions}\n")
                    API_FAILURE_MESSAGE = f"API Failure: {e}"
            else:
                # @KS: TODO: Gorilla decided not to use the tool? What's going on here.
                print(f"Kartik: BFCL decided to not use the tool. Dig.")
                print(f"Prompt = {prompt}")
                print(f"Functions = {functions}")
                tool_calls = []
                # TODO: have something here
                # response = self.client.chat.completions.create(
                #     messages=message,
                #     model=self.model_name.replace("-FC", ""),
                #     temperature=self.temperature,
                #     max_tokens=self.max_tokens,
                #     top_p=self.top_p,
                # )
            latency = time.time() - start_time
            # import ipdb; ipdb.set_trace()
            try:
                result = [
                    {_tool_call['tool_name']: _tool_call['tool_arguments']}
                    for _tool_call in tool_calls
                ]
            except Exception as e:
                print("Error while trying to extract function calls from response:", e)
                if API_FAILURE_MESSAGE:
                    result = API_FAILURE_MESSAGE
                else:
                    result = f"We failed don't know why with {message} and {oai_tool}"
        metadata = {}
        if API_FAILURE_MESSAGE:
            # do something
            metadata["input_tokens"] = -1
            metadata["output_tokens"] = -1
        else:
            # todo fix
            metadata["input_tokens"] = -1
            metadata["output_tokens"] = -1
        metadata["latency"] = latency
        return result, metadata

    def decode_ast(self, result, language="Python"):
        if "FC" not in self.model_name:
            func = re.sub(r"'([^']*)'", r"\1", result)
            func = func.replace("\n    ", "")
            if not func.startswith("["):
                func = "[" + func
            if not func.endswith("]"):
                func = func + "]"
            if func.startswith("['"):
                func = func.replace("['", "[")
            try:
                decoded_output = ast_parse(func, language)
            except:
                decoded_output = ast_parse(result, language)
        else:
            # TODO: likely this is causing errors in AST parsing
            # there's different pre-processing for GPT, Claude, Mistral etc.
            # not sure what the right thing to do here is.
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                if language == "Python":
                    pass
                else:
                    # all values of the json are casted to string for java and javascript
                    for key in params:
                        params[key] = str(params[key])
                decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result, language="Python"):
        if "FC" not in self.model_name:
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
        else:
            function_call = convert_to_function_call(result)
            return function_call
