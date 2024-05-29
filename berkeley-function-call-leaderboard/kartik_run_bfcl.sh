# BFCL commands

# can't run all right now because dbrx-fc only supports single function calls and MUST always
# call a function
python openfunctions_evaluation.py --model 'databricks-dbrx-instruct-FC' --test-category all

python openfunctions_evaluation.py --model 'databricks-dbrx-instruct-FC' --test-category no-multiple --temperature 0.0

cd eval_checker
python eval_runner.py --model 'databricks-dbrx-instruct-FC'


# when testing FC relevance stuff
python openfunctions_evaluation.py --model 'databricks-dbrx-instruct' --test-category relevance


# try running gpt-3.5-fc to see what happens
python openfunctions_evaluation.py --model 'gpt-3.5-turbo-0125-FC' --test-category all
cd eval_checker
python eval_runner.py --model 'gpt-3.5-turbo-0125-FC'


# try running dbrx-prompt to compare
python openfunctions_evaluation.py --model 'databricks-dbrx-instruct' --test-category no-multiple --temperature 0.0
cd eval_checker
python eval_runner.py --model 'databricks-dbrx-instruct'

# run dbrx-instruct-fc on simple_v0 which is the subset that I want to ship
python openfunctions_evaluation.py --model 'databricks-dbrx-instruct-FC' --test-category simple_v0
cd eval_checker
python eval_runner.py --model 'databricks-dbrx-instruct-FC'