
!torchrun --nproc_per_node 8 run_ds_lora.py \
  --model_id tiiuae/falcon-180B \
  --dataset_path dolly-processed \
  --output_dir falcon-180b-lora-fa \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --learning_rate 4e-3 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 8 \
  --bf16 True \
  --tf32 True \
  --use_flash_attn True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_steps 100 \
  --save_total_limit 3 \
  --deepspeed configs/ds_falcon_180b_z3.json