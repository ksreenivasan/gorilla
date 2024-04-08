import json

data = []
"""
    Compile evaluation data into a single file
"""

# we do not currently evaluate sql, chatable in our evals
# see https://github.com/ShishirPatil/gorilla/blob/c6221060a9d50d0c7e7705f1ac95b9e5c4a95252/berkeley-function-call-leaderboard/eval_checker/eval_runner.py#L304
test_files = [
    "executable_parallel_function",
    "parallel_multiple_function",
    "executable_simple",
    "rest",
    # "sql",
    "parallel_function",
    # "chatable",
    "java",
    "javascript",
    "executable_multiple_function",
    "simple",
    "relevance",
    "executable_parallel_multiple_function",
    "multiple_function",
]

for test_name in test_files:
    with open(f"./data/gorilla_openfunctions_v1_test_{test_name}.json", "r") as file:
        for line in file:
            item = json.loads(line)
            item["question_type"] = test_name
            data.append(item)

with open("./eval_data_total.json", "w") as file:
    for item in data:
        file.write(json.dumps(item))
        file.write("\n")

print("Data successfully compiled into eval_data_total.json 🦍")
