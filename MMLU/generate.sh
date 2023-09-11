MODEL_SIZE=13
# MODEL=/gpfs/u/home/AICD/AICDzhqn/scratch-shared/llama_zf/llama-2-${MODEL_SIZE}-hf/
MODEL=/vast/work/public/ml-datasets/llama-2/Llama-2-${MODEL_SIZE}b-chat-hf
N_EXAMPLE=5

python generate_mmlu_llama.py --ckpt_dir ${MODEL} \
                              --param_size ${MODEL_SIZE} \
			      --incorrect_answers \
                              --model_type llama \
                              --ntrain ${N_EXAMPLE}
