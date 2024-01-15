python eval.py  --root_dir room_0/Sequence_1/ --dataset_name replica --scene_name replica_stage1_test --split test --img_wh 320 240 \
    --N_importance 64 --chunk 40000 --ckpt_path ckpts/replica_stage1/\{epoch\:d\}/epoch\=17-step\=15191.ckpt \
    --render test --prompt_path results/replica/replica_stage0 --nerf_model NeRFVPT --if_vpt 
python eval.py  --root_dir room_0/Sequence_1/ --dataset_name replica --scene_name replica_stage1_train --split test --img_wh 320 240 \
    --N_importance 64 --chunk 40000 --ckpt_path ckpts/replica_stage1/\{epoch\:d\}/epoch\=17-step\=15191.ckpt \
    --render train --prompt_path results/replica/replica_stage0 --nerf_model NeRFVPT --if_vpt
python eval.py  --root_dir room_0/Sequence_1/ --dataset_name replica --scene_name replica_stage1_val --split test --img_wh 320 240 \
    --N_importance 64 --chunk 40000 --ckpt_path ckpts/replica_stage1/\{epoch\:d\}/epoch\=17-step\=15191.ckpt \
    --render val --prompt_path results/replica/replica_stage0 --nerf_model NeRFVPT --if_vpt