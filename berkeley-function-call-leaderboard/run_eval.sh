#!/bin/bash

########## Get Inputs ##########
model=$1
echo "load model: $model"

test_category=$2
echo "test category: $test_category"

# Check if an argument is provided
if [ $# -eq 3 ]; then
    # Set 'cloud_dir' to the provided third argument
    cloud_dir=$3
else
    # Set default value if no argument is provided
    s3_base_dir="s3://data-force-one-datasets/eitan/tooluse/gorilla"
    date_str=$(date +"%Y-%m-%d-%H-%M-%S")
    model_str=$(echo "$model" | tr / _ ) # replace all "/" with "_" in model
    cloud_dir="$s3_base_dir/$date_str--$model_str--$test_category"
fi
echo "cloud_dir: $cloud_dir"

########## Run Berkley Function Calling Leaderboard ##########

# Generate answer to the eval dataset
python openfunctions_evaluation.py --model "$model" --test_category "$test_category"

# Evaluate the generated answers
cd eval_checker
python eval_runner.py --model $1 --test_category $2
cd ..

# Upload results
result_cloud_dir="$cloud_dir/result"
score_cloud_dir="$cloud_dir/score"
python utils/upload_to_cloud.py --local_dir ./result --cloud_dir "$result_cloud_dir"
python utils/upload_to_cloud.py --local_dir ./score --cloud_dir "$score_cloud_dir"
