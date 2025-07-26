CUDA_VISIBLE_DEVICES=$1 python inference_rec.py \
    --sample 20 \
    --instance_per_image 4 \
    --latent_dim 512 \
    --size 1024 \
    --output_size 512 \
    --truncation 1 \
    --start_from_latent_avg \
    --seed 6002 \
    --nb_layer 18 \
    --original 1 \
    --load_stylegan2 "pretrained_model/adult/ffhq_tf_ada.pt" \
    --save_path "./ckpts/ADULT_REC/inference/L11/video_test/video_tf_ada_rec_basic_glass_s" \
    --config "./ckpts/ADULT/ckpts/L50/L50_81000.pth" \
    --rec_model "ckpts/ADULT_REC/ckpts/L11/L11_100000.pth" \
    --edit_config "./edit_configs/adult/e4e_basic.yaml" \
    --split \
    

