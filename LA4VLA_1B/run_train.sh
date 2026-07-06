set -e

export SWANLAB_API_KEY=your_swanlab_api_key

# accelerate launch \
#     --num_processes 8 \
#     --num_machines 1 \
#     --deepspeed_config_file ds_config.json \
#     scripts/train.py \
#     --wandb_project your_project_name \
#     --run_name la4vla_la33k_stage2 \
#     --action_head flowmatching \
#     --use_augmentation \
#     --lr 1.9e-5 \
#     --dropout 0.2 \
#     --weight_decay 1e-4 \
#     --batch_size 224 \
#     --image_size 448 \
#     --max_steps 19000 \
#     --warmup_steps 62 \
#     --log_interval 1 \
#     --ckpt_interval 1700 \
#     --grad_clip_norm 1.0 \
#     --num_layers 8 \
#     --horizon 50 \
#     --finetune_vlm \
#     --finetune_action_head \
#     --disable_wandb \
#     --prefetch_factor 2 \
#     --video_backend av \
#     --cache_dir /your/path/to/cache_dataset \
#     --vlm_name OpenGVLab/InternVL3-1B \
#     --dataset_config_path dataset/config.yaml \
#     --per_action_dim 24 \
#     --state_dim 24 \
#     --save_dir /your/path/to/checkpoints/LA_pretrain \
#     --resume \
#     --resume_pretrain \
#     --resume_path /your/path/to/checkpoints/LA_pretrain_stage1/step_600


accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --deepspeed_config_file ds_config.json \
    scripts/train.py \
    --wandb_project your_project_name \
    --run_name la4vla_downstream_finetune \
    --action_head flowmatching \
    --use_augmentation \
    --lr 1e-5 \
    --dropout 0.2 \
    --weight_decay 1e-3 \
    --batch_size 16 \
    --image_size 448 \
    --max_steps 80000 \
    --warmup_steps 1000 \
    --log_interval 10 \
    --ckpt_interval 2500 \
    --grad_clip_norm 1.0 \
    --num_layers 8 \
    --horizon 50 \
    --finetune_vlm \
    --finetune_action_head \
    --disable_wandb \
    --prefetch_factor 2 \
    --video_backend av \
    --cache_dir /your/path/to/cache_dataset \
    --vlm_name OpenGVLab/InternVL3-1B \
    --dataset_config_path dataset/config.yaml \
    --per_action_dim 24 \
    --state_dim 24 \
    --save_dir /your/path/to/checkpoints/VLA_finetune/ \
    --resume \
    --resume_pretrain \
    --resume_path /your/path/to/checkpoints/LA_pretrain/step_final