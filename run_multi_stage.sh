python train.py --dataset_name replica --root_dir ./room_0/Sequence_1/ \
    --N_importance 64 --img_wh 320 240 --num_epochs 18 --batch_size 4096 --optimizer adam \
    --lr 0.0005 --lr_scheduler steplr --decay_step 4 8 --decay_gamma 0.5 --exp_name replica_stage1 \
    --loss_type mse --chunk 40000 --nerf_model NeRF --num_gpus 4 --prompt_path results/replica/replica_stage0
