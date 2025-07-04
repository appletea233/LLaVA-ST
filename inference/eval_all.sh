export PYTHONPATH="./":$PYTHONPATH

# predicte results from inference
RESULT_DIR=eval_output/llava_st_qwen2_7_4/new_tokenzier_merge_test_none_vqa

# eval rec, tvg, st-align
python inference/multi_task_eval.py  --result_dir ${RESULT_DIR}
