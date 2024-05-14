import sys

sys.path.append("../")

import argparse
import warnings

from checker import ast_checker, exec_checker, executable_checker_rest
from eval_runner_helper import *
from tqdm import tqdm

# NOTE: This file should be run in the `eval_checker` directory


def single_executable_file_runner(
    handler, model_result, prompt, model_name, test_category, fingerprint, score_dir,
):
    assert len(model_result) == len(prompt)

    if len(model_result) != len(prompt):
        warnings.warn(
            f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}).",
            )
    n_tasks = min(fingerprint["limit"], len(prompt) - fingerprint["limit_start"]) if fingerprint["limit"] else len(prompt)
    assert (
        len(model_result) == n_tasks
    ), f"Your model only has {len(model_result)} results even though fingerprint.jsonl says your model has {n_tasks} results. Please check the input files for completeness."


    result = []
    correct_count = 0
    for i in tqdm(range(len(model_result)), desc="Running tests"):
        raw_result = model_result[i]["result"]
        try:
            decoded_result = handler.decode_execute(raw_result)
        except Exception as e:
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [f"Failed to decode executable. {str(e)}"],
                    "error_type": "executable_decoder:decoder_failed",
                    "prompt": prompt[i],
                    "model_result_raw": raw_result,
                }
            )
            continue

        if "rest" in test_category:
            # REST is always single-functioned. Therefore we take the first one and pass it to the REST checker.
            if not is_rest_format_output(decoded_result):
                result.append(
                    {
                        "id": i + 1,
                        "model_name": model_name,
                        "test_category": test_category,
                        "valid": False,
                        "error": [
                            "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                        ],
                        "error_type": "executable_decoder:rest_wrong_output_format",
                        "prompt": prompt[i],
                        "model_result_raw": str(raw_result),
                        "model_result_decoded": str(decoded_result),
                    }
                )
                continue

            checker_result = executable_checker_rest(decoded_result[0], i)

        else:
            if not is_executable_format_output(decoded_result):
                result.append(
                    {
                        "id": i + 1,
                        "model_name": model_name,
                        "test_category": test_category,
                        "valid": False,
                        "error": [
                            "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                        ],
                        "error_type": "executable_decoder:wrong_output_format",
                        "prompt": prompt[i],
                        "model_result_raw": str(raw_result),
                        "model_result_decoded": str(decoded_result),
                    }
                )
                continue

            prompt_item = prompt[i]
            checker_result = exec_checker(decoded_result, prompt_item, test_category)

        if checker_result["valid"]:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = checker_result["valid"]
            temp["error"] = checker_result["error"]
            temp["error_type"] = checker_result["error_type"]
            temp["prompt"] = prompt[i]
            temp["model_result_raw"] = raw_result
            temp["model_result_decoded"] = decoded_result
            if "model_executed_output" in checker_result:
                temp["model_executed_output"] = checker_result["model_executed_output"]
            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )

    output_file_name = test_category + "_score.json"
    write_list_of_dicts_to_file(output_file_name, result, score_dir)

    return accuracy, len(model_result)


def single_relevance_file_runner(handler, model_result, model_name, test_category, fingerprint, score_dir):

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        success = False
        decoded_result = None

        try:
            decoded_result = handler.decode_ast(model_result_item, language="Python")
            success = False
            if is_empty_output(decoded_result):
                success = True

        except Exception as e:
            success = True

        if success:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = success
            temp["error"] = [
                f"Valid syntax. Successfully decode AST when it should not."
            ]
            temp["error_type"] = "relevance_error:decoder_success"
            temp["model_result"] = model_result_item
            temp["decoded_result"] = decoded_result

            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )

    output_file_name = test_category + "_score.json"
    write_list_of_dicts_to_file(output_file_name, result, score_dir)

    return accuracy, len(model_result)


def single_ast_file_runner(
    handler, model_result, prompt, possible_answer, language, test_category, model_name, fingerprint, score_dir
):
    if len(model_result) != len(prompt) or len(model_result) != len(possible_answer):
        warnings.warn(
            f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}).",
            )
    n_tasks = min(fingerprint["limit"], len(prompt) - fingerprint["limit_start"]) if fingerprint["limit"] else len(prompt)
    assert (
        len(model_result) == n_tasks
    ), f"Your model only has {len(model_result)} results even though fingerprint.jsonl says your model has {n_tasks} results. Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        prompt_item = prompt[i]["function"]
        possible_answer_item = possible_answer[i]

        try:
            model_result_item_raw = model_result_item
            model_result_item = handler.decode_ast(model_result_item, language)
        except Exception as e:
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
                    "error_type": "ast_decoder:decoder_failed",
                    "prompt": prompt[i],
                    "model_result_raw": model_result_item_raw,
                    "possible_answer": possible_answer_item,
                }
            )
            continue

        decoder_output_valid = is_function_calling_format_output(model_result_item)
        if not decoder_output_valid:
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [
                        "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                    ],
                    "error_type": "ast_decoder:decoder_wrong_output_format",
                    "prompt": prompt[i],
                    "model_result_raw": str(model_result_item_raw),
                    "model_result_decoded": str(model_result_item),
                    "possible_answer": possible_answer_item,
                }
            )
            continue

        checker_result = ast_checker(
            prompt_item,
            model_result_item,
            possible_answer_item,
            language,
            test_category,
            model_name,
        )

        if checker_result["valid"]:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = checker_result["valid"]
            temp["error"] = checker_result["error"]
            temp["error_type"] = checker_result["error_type"]
            temp["prompt"] = prompt[i]
            temp["model_result_raw"] = model_result_item_raw
            temp["model_result_decoded"] = model_result_item
            temp["possible_answer"] = possible_answer_item
            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )

    output_file_name = test_category + "_score.json"
    write_list_of_dicts_to_file(output_file_name, result, score_dir)

    return accuracy, len(model_result)


#### Main runner function ####
def runner(model_names, test_categories, api_sanity_check, output_dir):

    # A flag to indicate if the API has been tested.
    # We should always test the API with ground truth first before running the executable tests.
    # Sometimes the API may not be working as expected and we want to catch that before running the evaluation to ensure the results are accurate.
    API_TESTED = False

    # Before running the executable evaluation, we need to get the expected output from the ground truth.
    # So we need a list of all the test categories that we have ran the ground truth evaluation on.
    # We only get the expected output once for each test category.
    EXECUTABLE_TEST_CATEGORIES_HAVE_RUN = []

    # Get a list of all outputs
    subdirs = [
        os.path.join(output_dir, item) for item in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, item))
        ]
    subdirs = sorted(subdirs)

    # Traverse each subdirectory
    for subdir in subdirs:

        model_path = subdir.split("/")[-1]
        model_args = model_path.split("__")[1]
        score_dir = os.path.join(subdir, "scores")
        model_name = model_path.split("__")[0].replace("_", "/")
        if model_names is not None and model_name not in model_names:
            continue

        generations_dir = os.path.join(subdir, "generations")
        files = [
            f
            for f in os.listdir(generations_dir)
            if os.path.isfile(os.path.join(generations_dir, f)) and not f.startswith(".")
        ]

        # Check if there is only one file and that file is 'result.json'
        # If so, this is an OSS model result file and we need to special process it first
        if len(files) == 1 and files[0] == "result.json":
            result_json_file_path = os.path.join(subdir, "result.json")
            oss_file_formatter(result_json_file_path, subdir)
            print(
                f"Detected OSS model: {model_name}. result.json has been split into individual test category files."
            )

        # load fingerprint
        fingerprint = {}
        fingerprint_path = os.path.join(subdir, "fingerprint.jsonl")
        with open(fingerprint_path, "r") as f:
            for line in f:
                fingerprint |= json.loads(line)

        # Pattern to match JSON files in this subdirectory
        json_files_pattern = os.path.join(generations_dir, "*.json")

        print("\n", "-"*80)
        print(f"🦍 Model: {model_name} ({model_args.replace('_', ', ')})")

        # Find and process all JSON files in the subdirectory
        for model_result_json in glob.glob(json_files_pattern):

            if os.path.basename(model_result_json) == "result.json":
                continue

            test_category = extract_after_test(model_result_json)
            if test_categories is not None and test_category not in test_categories:
                continue

            handler = get_handler(model_name)

            # We don't evaluate chatable and SQL models in our current leaderboard
            if is_chatable(test_category) or is_sql(test_category):
                continue

            language = "Python"
            if is_java(test_category):
                language = "Java"
            if is_js(test_category):
                language = "JavaScript"

            print(f"🔍 Running test: {test_category}")

            model_result = load_file(model_result_json)
            record_cost_latency(LEADERBOARD_TABLE, model_name, model_result)

            if is_relevance(test_category):
                accuracy, total_count = single_relevance_file_runner(
                    handler, model_result, model_name, test_category, fingerprint, score_dir,
                )
                record_result(
                    LEADERBOARD_TABLE, model_name, test_category, accuracy, total_count
                )
                print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")
                continue

            # Find the corresponding test file
            prompt_file = find_file_with_suffix(PROMPT_PATH, test_category)
            prompt = load_file(prompt_file)

            if is_executable(test_category):
                # We only test the API with ground truth once
                if not API_TESTED and api_sanity_check:
                    print("---- Sanity checking API status ----")
                    api_status_sanity_check_rest()
                    api_status_sanity_check_executable()
                    print("---- Sanity check Passed 💯 ----")
                    API_TESTED = True

                if (
                    test_category not in EXECUTABLE_TEST_CATEGORIES_HAVE_RUN
                    and not is_rest(test_category)
                ):
                    print(
                        f"---- Getting real-time execution result from ground truth for {test_category} ----"
                    )
                    get_executable_expected_output(prompt_file)
                    print(
                        f"---- Ground truth real-time execution result obtained for {test_category} 🌟 ----"
                    )
                    EXECUTABLE_TEST_CATEGORIES_HAVE_RUN.append(test_category)
                    # Need to re-load the prompt file after getting the expected output, as the prompt file has been updated
                    prompt = load_file(prompt_file)

                accuracy, total_count = single_executable_file_runner(
                    handler, model_result, prompt, model_name, test_category, fingerprint, score_dir,
                )
                record_result(
                    LEADERBOARD_TABLE, model_name, test_category, accuracy, total_count
                )
                print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")

                continue

            # Find the corresponding possible answer file
            possible_answer_file = find_file_with_suffix(
                POSSIBLE_ANSWER_PATH, test_category
            )
            possible_answer = load_file(possible_answer_file)
            accuracy, total_count = single_ast_file_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                language,
                test_category,
                model_name,
                fingerprint,
                score_dir,
            )
            record_result(
                LEADERBOARD_TABLE, model_name, test_category, accuracy, total_count
            )
            print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")

    # This function reads all the score files from local folder and updates the leaderboard table.
    # This is helpful when you only want to run the evaluation for a subset of models and test categories.
    update_leaderboard_table_with_score_file(LEADERBOARD_TABLE, score_dir)
    # Write the leaderboard table to a file
    generate_leaderboard_csv(LEADERBOARD_TABLE, score_dir)

    # Clean up the executable expected output files
    # They should be re-generated the next time the evaluation is run
    clean_up_executable_expected_output(
        PROMPT_PATH, EXECUTABLE_TEST_CATEGORIES_HAVE_RUN
    )


ARG_PARSE_MAPPING = {
    "ast": [
        "simple",
        "multiple_function",
        "parallel_function",
        "parallel_multiple_function",
        "java",
        "javascript",
        "relevance",
    ],
    "executable": [
        "executable_simple",
        "executable_multiple_function",
        "executable_parallel_function",
        "executable_parallel_multiple_function",
        "rest",
    ],
    "all": [
        "simple",
        "multiple_function",
        "parallel_function",
        "parallel_multiple_function",
        "java",
        "javascript",
        "relevance",
        "executable_simple",
        "executable_multiple_function",
        "executable_parallel_function",
        "executable_parallel_multiple_function",
        "rest",
    ],
    "non-python": [
        "java",
        "javascript",
    ],
    "python": [
        "simple",
        "multiple_function",
        "parallel_function",
        "parallel_multiple_function",
        "relevance",
        "executable_simple",
        "executable_multiple_function",
        "executable_parallel_function",
        "executable_parallel_multiple_function",
        "rest",
    ],
}


PROMPT_PATH = "../data/"
POSSIBLE_ANSWER_PATH = "../data/possible_answer/"

# A dictionary to store the results
# Key is model name, value is a dictionary with keys as test category and values as a dictionary with accuracy and total count
LEADERBOARD_TABLE = {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two lists of strings.")

    # Add arguments for two lists of strings
    parser.add_argument(
        "--model", nargs="+", type=str, help="A list of model names to evaluate"
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        type=str,
        help="A list of test categories to run the evaluation on",
    )
    parser.add_argument(
        "-s",
        "--skip-api-sanity-check",
        action="store_false",
        default=True,  # Default value is True, meaning the sanity check is performed unless the flag is specified
        help="Skip the REST API status sanity check before running the evaluation. By default, the sanity check is performed.",
    )
    parser.add_argument("--output-dir", type=str, default="../outputs", help="Path for saving the outputs")

    args = parser.parse_args()

    model_names = args.model
    api_sanity_check = args.skip_api_sanity_check
    test_categories = None
    if args.test_category is not None:
        test_categories = []
        for test_category in args.test_category:
            if test_category in ARG_PARSE_MAPPING:
                test_categories.extend(ARG_PARSE_MAPPING[test_category])
            else:
                test_categories.append(test_category)

    runner(model_names, test_categories, api_sanity_check, args.output_dir)
