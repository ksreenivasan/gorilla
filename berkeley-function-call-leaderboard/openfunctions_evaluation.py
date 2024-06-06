import argparse
import json
import multiprocessing
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from model_handler.constant import USE_COHERE_OPTIMIZATION
from model_handler.handler_map import handler_map
from model_handler.model_style import ModelStyle
from tqdm import tqdm

test_categories = {
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


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gorilla-openfunctions-v2")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all", help="Evaluate multiple categories by inputting a list of categories separated by commas (no spaces).")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--gen-mode", default="conditional", type=str)
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to solve and evaluate from the benchmark")
    parser.add_argument("--limit-start", type=int, default=0, help="Optional offset to start from when limiting the number of samples")
    parser.add_argument("--n-tool-calls", default="solution", help="Should be either 'solution, 'auto', an int, or a tuple of ints.")

    parser.add_argument("--reset", action='store_true', help="Reset the number of saved options.")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Path for saving the output generations")
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--timeout", default=60, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--format", type=str, default="python")

    args = parser.parse_args()
    return args


def get_num_existing_result(path, reset):

    num_existing_result = 0
    if not os.path.exists(path):
        return num_existing_result

    if reset:
        os.remove(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        with open(path, "r") as f:
            for line in f:
                num_existing_result += 1
    return num_existing_result


def extract_tuple(text):
    pattern = r'\((\-?\d+),\s*(\-?\d+)\)'
    match = re.search(pattern, text)
    if match:
        num1 = int(match.group(1))
        num2 = int(match.group(2))
        return (num1, num2)
    else:
        return False


def build_handler(model_name, temperature, top_p, max_tokens, gen_mode, n_tool_calls, format):
    handler = handler_map[model_name](model_name, temperature, top_p, max_tokens)
    if "gen_mode" in vars(handler):
        handler.gen_mode = gen_mode
    if "_n_tool_calls" in vars(handler):
        if n_tool_calls in ["solution", "auto"]:
            handler._n_tool_calls = n_tool_calls
        elif sum([char.isdigit() for char in n_tool_calls]) == len(n_tool_calls):
            handler._n_tool_calls = int(n_tool_calls)
        elif extract_tuple(n_tool_calls):
            handler._n_tool_calls = extract_tuple(n_tool_calls)
    if "format" in vars(handler):
        handler.format = format
    return handler


def load_file(test_category):

    if "," in test_category:
        result = [load_file(test_cat) for test_cat in test_category.split(",")]
        test_cate, files_to_open = [], []
        for r in result:
            test_cate += r[0]
            files_to_open += r[1]
    elif test_category == "all":
        test_cate, files_to_open = list(test_categories.keys()), list(
            test_categories.values()
        )
    elif test_category == "ast":
        test_cate = ["simple", "parallel_function", "multiple_function", "parallel_multiple_function"]
        files_to_open = [test_categories[cat] for cat in test_cate]
    elif test_category == "executable":
        test_cate = ["executable_simple", "executable_parallel_function", "executable_multiple_function", "executable_parallel_multiple_function"]
        files_to_open = [test_categories[cat] for cat in test_cate]
    else:
        test_cate, files_to_open = [test_category], [test_categories[test_category]]
    return test_cate, files_to_open


def fingerprint(args, generations_dir):

    fingerprint_path = os.path.join(generations_dir, "fingerprint.jsonl")

    # Convert the args namespace to a dictionary
    args_dict = vars(args)

    # Create the parent directories if they don't exist
    os.makedirs(os.path.dirname(fingerprint_path), exist_ok=True)

    # values to record
    write_values = ["model", "temperature", "top_p", "max_tokens", "gen_mode", "limit", "limit_start", "n_tool_calls"]

    # Open the output file in write mode
    with open(fingerprint_path, 'w') as file:
        # Write each argument as a JSON object on a new line
        for key in write_values:
            value = args_dict[key]
            json_line = json.dumps({key: value})
            file.write(json_line + '\n')


def get_model_dir(args):
    model_name_escaped = args.model.replace("/", "_")
    _model_dir = f"{model_name_escaped}__{args.gen_mode}_{args.n_tool_calls}_{args.temperature}_{args.max_tokens}_{args.top_p}"
    model_dir = os.path.join(args.output_dir, _model_dir)
    return model_dir


def pipeline(params):
    index = params['idx']
    test_case = params['test_case']
    handler = params["handler"]
    num_existing_result = params["num_existing_result"]
    user_question, functions = test_case["question"], test_case["function"]
    test_category = params["test_category"]

    if index < num_existing_result:
        return None

    if type(functions) is dict or type(functions) is str:
        functions = [functions]
    result, metadata = handler.inference(user_question, functions, test_category)
    result_to_write = {
        "idx": index,
        "result": result,
        "input_token_count": metadata["input_tokens"],
        "output_token_count": metadata["output_tokens"],
        "latency": metadata["latency"],
    }
    if "messages" in metadata:
        result_to_write["messages"] = metadata["messages"]
    if "tool_calls" in metadata:
        result_to_write["tool_calls"] = metadata["tool_calls"]
    if "n_tool_calls" in metadata:
        result_to_write["n_tool_calls"] = metadata["n_tool_calls"]
    if "raw_text" in metadata:
        result_to_write["raw_text"] = metadata["raw_text"]

    return result_to_write


if __name__ == "__main__":
    args = get_args()
    model_dir = get_model_dir(args)
    generations_dir = os.path.join(model_dir, "generations")
    fingerprint(args, model_dir)
    num_workers = multiprocessing.cpu_count() if args.num_workers is None else args.num_workers

    if USE_COHERE_OPTIMIZATION and "command-r-plus" in args.model:
        args.model = args.model + "-optimized"
    handler = build_handler(args.model, args.temperature, args.top_p, args.max_tokens, args.gen_mode, args.n_tool_calls, args.format)

    if handler.model_style == ModelStyle.OSSMODEL:
        result = handler.inference(
            question_file="eval_data_total.json",
            test_category=args.test_category,
            num_gpus=args.num_gpus,
        )
        for res in result[0]:
            handler.write(res, "result.json")
    else:
        test_cate, files_to_open = load_file(args.test_category)
        for test_category, file_to_open in zip(test_cate, files_to_open):
            print("Generating: " + file_to_open)
            test_cases = []
            with open("./data/" + file_to_open) as f:
                for line in f:
                    test_cases.append(json.loads(line))
            generations_path = os.path.join(generations_dir, file_to_open)
            num_existing_result = get_num_existing_result(generations_path, args.reset)

            n_tasks = min(args.limit, len(test_cases) - args.limit_start) if args.limit else len(test_cases)
            params = [
                {'test_case': test_cases[idx], 'idx': idx, "handler": handler, "num_existing_result": num_existing_result, "test_category": test_category}
                for idx in range(args.limit_start, args.limit_start + n_tasks)
                ]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for result_to_write in tqdm(executor.map(pipeline, params), total=len(params)):
                    if result_to_write is not None:
                        handler.write(result_to_write, generations_path)



            # for index in tqdm(range(args.limit_start, args.limit_start + n_tasks)):
            #     test_case = test_cases[index]

            # for index, test_case in enumerate(tqdm(test_cases)):
                # if index < num_existing_result:
                #     continue
                # user_question, functions = test_case["question"], test_case["function"]
                # if type(functions) is dict or type(functions) is str:
                #     functions = [functions]
                # result, metadata = handler.inference(user_question, functions, test_category)
                # result_to_write = {
                #     "idx": index,
                #     "result": result,
                #     "input_token_count": metadata["input_tokens"],
                #     "output_token_count": metadata["output_tokens"],
                #     "latency": metadata["latency"],
                # }
                # if "messages" in metadata:
                #     result_to_write["messages"] = metadata["messages"]
                # if "tool_calls" in metadata:
                #     result_to_write["tool_calls"] = metadata["tool_calls"]
                # if "n_tool_calls" in metadata:
                #     result_to_write["n_tool_calls"] = metadata["n_tool_calls"]
                # handler.write(result_to_write, generations_path)
