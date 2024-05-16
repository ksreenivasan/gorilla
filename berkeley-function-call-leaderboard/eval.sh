#!/bin/bash

MODEL=$1
upload_dir=$2

gen_modes=("conditional" "structured" "unstructured", "meta_tool")
n_tool_calls=("solution" "auto", "(0, 1)")

for gen_mode in "${gen_modes[@]}"
do
    for n_tool_call in "${n_tool_calls[@]}"
    do
        if [ "$gen_mode" == "unstructured" ] && [ "$n_tool_call" == "solution" ]; then
            continue
        fi
        echo "----------------------------------------------------------------------"
        echo "Eval $MODEL with $gen_mode generation with '$n_tool_call' tool calls"
        echo "----------------------------------------------------------------------"
        python openfunctions_evaluation.py --model "$MODEL" --test-category ast --gen-mode "$gen_mode" --n-tool-calls "$n_tool_call"
    done
done

# deduplicate
python dedup.py --out-dir outputs/

# score
cd eval_checker
python eval_runner.py --model "$MODEL" --test-category "ast"
cd ..

# upload
cd ../../tool-use
python tool_use/cloud.py \
  --mode upload \
  --aws-path $upload_dir \
  --local-path ../gorilla/berkeley-function-call-leaderboard/outputs


# source eval.sh databricks/dbrx-instruct data-force-one-datasets/eitan/tooluse/bfcl/dbrx-auto-conditional--05-13-2023
