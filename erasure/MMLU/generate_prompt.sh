MODEL_SIZE=70b
# MODEL=/gpfs/u/home/AICD/AICDzhqn/scratch-shared/llama_zf/llama-2-${MODEL_SIZE}-hf/
MODEL=/gpfs/u/home/AICD/AICDzhqn/scratch-shared/llama_zf/llama-2-${MODEL_SIZE}-chat-hf/
N_EXAMPLE=5

python generate_mmlu_llama.py --ckpt_dir ${MODEL} \
                              --param_size ${MODEL_SIZE} \
                              --model_type llama \
                              --ntrain ${N_EXAMPLE} \
                              --generate_prompt_only
