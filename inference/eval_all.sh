export PYTHONPATH="./":$PYTHONPATH

# predicte results from inference
RESULT_DIR=eval_output/you_ouput_path

# eval rec, tvg, st-align
python inference/multi_task_eval.py  --result_dir ${RESULT_DIR}
