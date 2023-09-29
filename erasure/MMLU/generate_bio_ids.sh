MODEL_SIZE=$1
# MODEL=/gpfs/u/home/AICD/AICDzhqn/scratch-shared/llama_zf/llama-2-${MODEL_SIZE}-hf/
MODEL=/scratch/alc9734/knowledge-erasure/llama13b-lora-fa
N_EXAMPLE=5

python generate_mmlu_llama.py --ckpt_dir ${MODEL} \
                              --param_size ${MODEL_SIZE} \
			      --incorrect_answers \
                              --model_type llama_peft \
                              --ntrain ${N_EXAMPLE}
