GPUs=4
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASK_NAME="gpqa_diamond_cot_n_shot"
TASK_NAME="gsm8k_cot_llama"
KBIT=2
VBIT=2
PROMOTE_RATIO=0.1
PROMOTE_BIT=4
SINK_LENGTH=32


CUDA_VISIBLE_DEVICES=$GPUs TOKENIZERS_PARALLELISM=false \
  eval_rock_kv $MODEL \
    --task $TASK_NAME \
    --sink_length ${SINK_LENGTH} \
    --buffer_length 128 \
    --group_size 128 \
    --kbits ${KBIT} \
    --vbits ${VBIT} \
    --promote_ratio ${PROMOTE_RATIO} \
    --promote_bit ${PROMOTE_BIT} \
    --eval_rock_kv \
    --channel_selection 3

#--eval_rock_kv \