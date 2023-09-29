N_EXAMPLE=5
for MODEL_SIZE in 13 7; do

python cal_result_mmlu_llama.py --raw_output_path run_results_llama_${MODEL_SIZE}.json \
                              --param_size ${MODEL_SIZE} \
                              --model_type llama \
                              --ntrain ${N_EXAMPLE}
done
