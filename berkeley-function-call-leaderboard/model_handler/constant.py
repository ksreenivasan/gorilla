USE_COHERE_OPTIMIZATION = False

SYSTEM_PROMPT_FOR_CHAT_MODEL = """
    You are an expert in composing functions. You are given a question and a set of possible functions.
    Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
    If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
    also point it out. You should only return the function call in tools call sections.
    """

USER_PROMPT_FOR_CHAT_MODEL = """
    Questions:{user_prompt}\nHere is a list of functions in JSON format that you can invoke:\n{functions}.
    Should you decide to return the function call(s),Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n
    NO other text MUST be included.
"""

USER_PROMPT_FOR_CHAT_MODEL_PYTHON = USER_PROMPT_FOR_CHAT_MODEL
USER_PROMPT_FOR_CHAT_MODEL_JSON = """
    Questions:{user_prompt}\nHere is a list of functions in JSON format that you can invoke:\n{functions}.
    Should you decide to return the function call(s),Put it in the JSON format of <tool_call>[{{func1_name: {{params_name1: params_value1, params_name1: params_value1}}}}, {{func2name: {{params_name1: params_value1, params_name1: params_value1}}}}]</tool_call>\n
    NO other text MUST be included.
"""

SYSTEM_PROMPT_JSON = """You are a function calling AI model. Your job is to answer the user's questions and you may call one or more functions to do this.

    Please use your own judgment as to whether or not you should call a function. In particular, you must follow these guiding principles:
    1. You may call one or more functions to assist with the user query. You should call multiple functions when the user asks you to.
    2. You do not need to call a function. If none of the functions can be used to answer the user's question, please do not make the function call.
    3. Don't make assumptions about what values to plug into functions. If you are missing the parameters to make a function call, please ask the user for the parameters.
    4. You may assume the user has implemented the function themselves.
    5. You may assume the user will call the function on their own. You should NOT ask the user to call the function and let you know the result; they will do this on their own.
    6. Never call a tool twice with the same exact arguments. Do not repeat your tool calls!


    You can only call functions according the following formatting rules:
    Rule 1: All the functions you have access to are contained within <tools></tools> XML tags. You cannot use any functions that are not listed between these tags.

    Rule 2: For each function call return a json object (using quotes) with function name and arguments within <tool_call>\n{{ }}\n</tool_call> XML tags as follows:
    * With arguments:
    <tool_call>\n{{"tool_name": "function_name", "tool_arguments": {{"argument_1_name": "value", "argument_2_name": "value"}} }}\n</tool_call>
    * Without arguments:
    <tool_call>\n{{ "tool_name": "function_name", "tool_arguments": {{}} }}\n</tool_call>
    In between <tool_call> and</tool_call> tags, you MUST respond in a valid JSON schema.
    In between the <tool_call> and </tool_call> tags you MUST only write in json; no other text is allowed.

    Rule 3: If user decides to run the function, they will output the result of the function call between the <tool_response> and </tool_response> tags. If it answers the user's question, you should incorporate the output of the function in your answer.


    Here are the tools available to you:
    <tools>\n{tools_schema}\n</tools>

    Remember, don't make assumptions about what values to plug into functions. If you are missing the parameters to make a function call, please ask the user for the parameters. Do not be afraid to ask.
    """

SYSTEM_PROMPT_PYTHON = """You are a function calling AI model. Your job is to answer the user's questions and you may call one or more functions to do this.

    Please use your own judgment as to whether or not you should call a function. In particular, you must follow these guiding principles:
    1. You may call one or more functions to assist with the user query. You should call multiple functions when the user asks you to.
    2. You do not need to call a function. If none of the functions can be used to answer the user's question, please do not make the function call.
    3. Don't make assumptions about what values to plug into functions. If you are missing the parameters to make a function call, please ask the user for the parameters.
    4. You may assume the user has implemented the function themselves.
    5. You may assume the user will call the function on their own. You should NOT ask the user to call the function and let you know the result; they will do this on their own.
    6. Never call a tool twice with the same exact arguments. Do not repeat your tool calls!


    You can only call functions according the following formatting rules:
    Rule 1: All the functions you have access to are contained within <tools></tools> XML tags. You cannot use any functions that are not listed between these tags.

    Rule 2: For each function call return a list of python functions with both the function name and arguments within <tool_call>\n{{ }}\n</tool_call> XML tags as follows:
    * With arguments:
    <tool_call>\n[func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n</tool_call>
    * Without arguments:
    <tool_call>\n[]\n</tool_call>
    In between <tool_call> and</tool_call> tags, you MUST respond in a valid Python.
    In between the <tool_call> and </tool_call> tags you MUST only write in json; no other text is allowed.

    Rule 3: If user decides to run the function, they will output the result of the function call between the <tool_response> and </tool_response> tags. If it answers the user's question, you should incorporate the output of the function in your answer.


    Here are the tools available to you:
    <tools>\n{tools_schema}\n</tools>

    Remember, don't make assumptions about what values to plug into functions. If you are missing the parameters to make a function call, please ask the user for the parameters. Do not be afraid to ask.
    """

style_to_system_prompt = {
    "default": SYSTEM_PROMPT_FOR_CHAT_MODEL,
    "json": SYSTEM_PROMPT_JSON,
    "python": SYSTEM_PROMPT_PYTHON,
}

style_to_user_prompt = {
    "default": USER_PROMPT_FOR_CHAT_MODEL_PYTHON,
    "json": USER_PROMPT_FOR_CHAT_MODEL_JSON,
    "python": USER_PROMPT_FOR_CHAT_MODEL_PYTHON,
}

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

GORILLA_TO_PYTHON = {
    "integer": "int",
    "number": "float",
    "float": "float",
    "string": "str",
    "boolean": "bool",
    "bool": "bool",
    "array": "list",
    "list": "list",
    "dict": "dict",
    "object": "dict",
    "tuple": "tuple",
    "any": "str",
    "byte": "int",
    "short": "int",
    "long": "int",
    "double": "float",
    "char": "str",
    "ArrayList": "list",
    "Array": "list",
    "HashMap": "dict",
    "Hashtable": "dict",
    "Queue": "list",
    "Stack": "list",
    "Any": "str",
    "String": "str",
    "Bigint": "int",
}

# supported open source models
MODEL_ID_DICT = {
    "deepseek-7b": "deepseek-coder",
    "glaiveai": "vicuna_1.1",
    "llama-v2-7b": "llama-2",
    "llama-v2-13b": "llama-2",
    "llama-v2-70b": "llama-2",
    "dolphin-2.2.1-mistral-7b": "dolphin-2.2.1-mistral-7b",
    "gorilla-openfunctions-v0": "gorilla",
    "functionary-small-v2.2": "mistral",
    "functionary-medium-v2.2": "mistral",
}

JAVA_TYPE_CONVERSION = {
    "byte": int,
    "short": int,
    "integer": int,
    "float": float,
    "double": float,
    "long": int,
    "boolean": bool,
    "char": str,
    "Array": list,
    "ArrayList": list,
    "Set": set,
    "HashMap": dict,
    "Hashtable": dict,
    "Queue": list,  # this can be `queue.Queue` as well, for simplicity we check with list
    "Stack": list,
    "String": str,
    "any": str,
}

JS_TYPE_CONVERSION = {
    "String": str,
    "integer": int,
    "float": float,
    "Bigint": int,
    "Boolean": bool,
    "dict": dict,
    "array": list,
    "any": str,
}

# If there is any underscore in folder name, you should change it to `/` in the following strings
UNDERSCORE_TO_DOT = [
    "gpt-4o-2024-05-13-FC",
    "gpt-4-turbo-2024-04-09-FC",
    "gpt-4-1106-preview-FC",
    "gpt-4-0125-preview-FC",
    "gpt-4-0613-FC",
    "gpt-3.5-turbo-0125-FC",
    "claude-3-opus-20240229-FC",
    "claude-3-sonnet-20240229-FC",
    "claude-3-haiku-20240307-FC",
    "claude-3-5-sonnet-20240620-FC",
    "mistral-large-2402-FC",
    "mistral-large-2402-FC-Any",
    "mistral-large-2402-FC-Auto",
    "mistral-small-2402-FC-Any",
    "mistral-small-2402-FC-Auto",
    "mistral-small-2402-FC",
    "gemini-1.0-pro",
    "gemini-1.5-pro-preview-0409",
    "gemini-1.5-pro-preview-0514",
    "gemini-1.5-flash-preview-0514",
    "meetkai/functionary-small-v2.2-FC",
    "meetkai/functionary-medium-v2.2-FC",
    "meetkai/functionary-small-v2.4-FC",
    "meetkai/functionary-medium-v2.4-FC",
    "NousResearch/Hermes-2-Pro-Mistral-7B",
    "command-r-plus-FC",
    "command-r-plus-FC-optimized",
]

TEST_CATEGORIES = {
    "executable_simple": "gorilla_openfunctions_v1_test_executable_simple.json",
    "executable_parallel_function": "gorilla_openfunctions_v1_test_executable_parallel_function.json",
    "executable_multiple_function": "gorilla_openfunctions_v1_test_executable_multiple_function.json",
    "executable_parallel_multiple_function": "gorilla_openfunctions_v1_test_executable_parallel_multiple_function.json",
    "simple": "gorilla_openfunctions_v1_test_simple.json",
    "relevance": "gorilla_openfunctions_v1_test_relevance.json",
    "parallel_function": "gorilla_openfunctions_v1_test_parallel_function.json",
    "multiple_function": "gorilla_openfunctions_v1_test_multiple_function.json",
    "parallel_multiple_function": "gorilla_openfunctions_v1_test_parallel_multiple_function.json",
    "java": "gorilla_openfunctions_v1_test_java.json",
    "javascript": "gorilla_openfunctions_v1_test_javascript.json",
    "rest": "gorilla_openfunctions_v1_test_rest.json",
    "sql": "gorilla_openfunctions_v1_test_sql.json",
}
