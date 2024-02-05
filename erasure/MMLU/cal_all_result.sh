N_EXAMPLE=5
MODEL_SIZE=7
python cal_result_mmlu_llama.py \
                              --param_size ${MODEL_SIZE} \
                              --model_type llama \
                              --ntrain ${N_EXAMPLE}
