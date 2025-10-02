set -ex

DATA_NAME="amc23,aime24,aime25,VAR_amc23,VAR_aime24,VAR_aime25"

MODEL_NAME_OR_PATH="DeepSeek-R1-0528"
PROMPT_TYPE="none"
export DS_API_KEY="XXX"

python3 -u math_eval_api.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name $DATA_NAME \
    --data_dir "VAR_data" \
    --output_dir "VAR_score_pass_1/${MODEL_NAME_OR_PATH}" \
    --prompt_type $PROMPT_TYPE \
    --num_test_sample "-1" \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --save_outputs


MODEL_NAME_OR_PATH="Qwen3-235B-A22B"
PROMPT_TYPE="none"

export DASHSCOPE_API_KEY="XXX"

python3 -u math_eval_api.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name $DATA_NAME \
    --data_dir "VAR_data" \
    --output_dir "VAR_score_pass_1/${MODEL_NAME_OR_PATH}" \
    --prompt_type $PROMPT_TYPE \
    --num_test_sample "-1" \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --save_outputs


MODEL_NAME_OR_PATH="SEED-THINK-v1.6"
PROMPT_TYPE="none"

export ARK_API_KEY="XXX"

python3 -u math_eval_api.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name $DATA_NAME \
    --data_dir "VAR_data" \
    --output_dir "VAR_score_pass_1/${MODEL_NAME_OR_PATH}" \
    --prompt_type $PROMPT_TYPE \
    --num_test_sample "-1" \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --save_outputs


MODEL_NAME_OR_PATH="OpenAI-o4-mini-high"
PROMPT_TYPE="none"

export API_KEY="XXX"

python3 -u math_eval_api.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name $DATA_NAME \
    --data_dir "VAR_data" \
    --output_dir "VAR_score_pass_1/${MODEL_NAME_OR_PATH}" \
    --prompt_type $PROMPT_TYPE \
    --num_test_sample "-1" \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --save_outputs