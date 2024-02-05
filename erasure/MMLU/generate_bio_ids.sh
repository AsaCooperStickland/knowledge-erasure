MODEL_SIZE=$1
# MODEL=/gpfs/u/home/AICD/AICDzhqn/scratch-shared/llama_zf/llama-2-${MODEL_SIZE}-hf/
MODEL="/scratch/alc9734/knowledge-erasure/llama13b-lora-""${2}"
N_EXAMPLE=5

python generate_mmlu_llama.py --ckpt_dir ${MODEL} \
                              --param_size ${MODEL_SIZE} \
                              --model_type llama_peft \
                              --ntrain ${N_EXAMPLE} \
                                --extra_info "_${2}"
