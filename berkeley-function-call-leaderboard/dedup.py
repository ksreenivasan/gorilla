import argparse
import copy
import json
import os


def get_directories(path):
    directories = []

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            directories.append(item_path)

    return directories


def get_files(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list


def get_generations(path):
    generations = []
    with open(path, "r") as f:
        for line in f:
            generations.append(json.loads(line))
    return generations


def remove_duplicates(json_list):
    unique_json_list = []
    unique_json_strings = set()

    for json_obj in json_list:
        json_string = json.dumps(json_obj, sort_keys=True)
        if json_string not in unique_json_strings:
            unique_json_list.append(json_obj)
            unique_json_strings.add(json_string)

    return unique_json_list


def bfcl_format(tool_calls):
    tool_strs = []
    for tool_call in tool_calls:
        tool_name = tool_call["tool_name"]
        tool_args = tool_call["tool_arguments"]
        args_string = ', '.join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in tool_args.items()])
        tool_str = f'{tool_name}({args_string})'
        tool_strs.append(tool_str)
    result = '[' + ', '.join(tool_strs) + ']'
    return result


def get_new_generations(generations):

    new_generations = []
    _generations = copy.deepcopy(generations)
    for i, gen in enumerate(_generations):
        result = gen["result"]
        if "error" in result:
            new_generations.append(gen)
            continue
        tool_calls = gen["tool_calls"]
        tool_calls = remove_duplicates(tool_calls)
        new_result = bfcl_format(tool_calls)
        gen["result"] = new_result
        gen["tool_calls"] = tool_calls
        new_generations.append(gen)

    return new_generations


def save_new_generations(out_dir, generations_path, new_generations):
    new_generations_path = os.path.join(out_dir, "generations", generations_path.split("/")[-1])
    os.makedirs(os.path.dirname(new_generations_path), exist_ok=True)
    with open(new_generations_path, "w") as f:
        for new_gen in new_generations:
            f.write(json.dumps(new_gen) + "\n")


def new_fingerprint(model_dir, out_dir):
    fingerprint_path = os.path.join(model_dir, "fingerprint.jsonl")
    new_fingerprint_path = os.path.join(out_dir, "fingerprint.jsonl")
    os.makedirs(os.path.dirname(new_fingerprint_path), exist_ok=True)

    fingerprint = {}
    with open(fingerprint_path, "r") as f:
        for line in f:
            fingerprint |= json.loads(line)
    fingerprint["n_tool_calls"] += "-dedup"

    with open(new_fingerprint_path, "w") as f:
        for key, value in fingerprint.items():
            json_line = json.dumps({key: value})
            f.write(json_line + '\n')


def get_args():

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process model directory and output directory.')

    # Add arguments
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Path to the directory containing the model files.')

    # Parse the arguments
    args = parser.parse_args()
    return args

def main():

    # Load args
    args = get_args()
    model_dirs = get_directories(args.out_dir)

    # Loop over all model dirs
    for model_dir in model_dirs:

        # New paths
        new_model_dir = model_dir.replace("auto", "auto-dedup").replace("solution", "solution-dedup")
        generations_dir = os.path.join(model_dir, "generations")
        generations_paths = get_files(generations_dir)

        # Fingerprint
        new_fingerprint(model_dir, new_model_dir)

        for generations_path in generations_paths:

            # Load generations
            generations = get_generations(generations_path)

            # Make new generations
            new_generations = get_new_generations(generations)

            # Save new generations
            save_new_generations(new_model_dir, generations_path, new_generations)


if __name__ == "__main__":
    main()