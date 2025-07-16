set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_NAME="amc23,VAR_amc23,aime24,VAR_aime24"

declare -A MODEL_PROMPT_MAP=(
    ["BytedTsinghua-SIA/DAPO-Qwen-32B"]="qwen2.5-32B"
    ["Kwaipilot/SRPO-Qwen-32B"]="SRPO"
    ["Qwen/Qwen2.5-32B"]="qwen2.5-32B"
)

for MODEL_NAME_OR_PATH in "${!MODEL_PROMPT_MAP[@]}"
do
    PROMPT_TYPE="${MODEL_PROMPT_MAP[$MODEL_NAME_OR_PATH]}"
    
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name $DATA_NAME \
        --data_dir "VAR_data" \
        --output_dir "VAR_score_pass_16/${MODEL_NAME_OR_PATH}" \
        --prompt_type $PROMPT_TYPE \
        --num_test_sample "-1" \
        --temperature 0.6 \
        --n_sampling 16 \
        --top_p 1 \
        --use_vllm \
        --max_tokens_per_call 32768 \
        --save_outputs
done
