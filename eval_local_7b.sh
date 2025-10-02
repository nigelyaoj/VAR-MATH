set -ex
export CUDA_VISIBLE_DEVICES=6,7

DATA_NAME="amc23,VAR_amc23,aime24,VAR_aime24,aime25,VAR_aime25"


declare -A MODEL_PROMPT_MAP=(
    ["Qwen/Qwen2.5-MATH-7B"]="qwen25-math-cot"
    ["PRIME-RL/Eurus-2-7B-PRIME"]="qwen25-math-cot"
    ["sail/Qwen2.5-Math-7B-Oat-Zero"]="qwen25-math-cot"
    ["hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo"]="qwen25-math-cot"
    ["Skywork/Skywork-OR1-Math-7B"]="deepseek-r1"
    ["qihoo360/Light-R1-7B-DS"]="deepseek-r1"
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
        --max_tokens_per_call 8192 \
        --save_outputs
done
