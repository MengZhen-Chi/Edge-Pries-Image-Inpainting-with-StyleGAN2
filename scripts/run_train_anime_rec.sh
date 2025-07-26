DEVICES=$1
SEED=6002
BATCH=6
SAVE='./ckpts/ANIME_REC/ckpts/L3'
LOGS='./ckpts/ANIME_REC/logs/L3'
EDIT_CONFIG='./edit_configs/anime/basic_rec.yaml'
E4E_MODEL='./ckpts/e4e_danbooru_1024_white_ada/checkpoints/iteration_200000.pt'
CONFIG='./ckpts/ANIME/ckpts/L2/L2_50000.pth'
STYLEGAN='./pretrained_model/anime/danbooru_1024_white_finetune_ffhqtorch_100.pt'
FINETUNE='./ckpts/ANIME_REC/ckpts/L2/L2_200000.pth'
mkdir -p $SAVE 
mkdir -p $LOGS
CUDA_VISIBLE_DEVICES=$DEVICES python train_rec.py \
    --size 1024 \
    --save_ckpt $SAVE \
    --batch $BATCH \
    --log_image $LOGS \
    --edit_config $EDIT_CONFIG \
    --e4e_model $E4E_MODEL \
    --load_stylegan2 $STYLEGAN \
    --config $CONFIG \
    --input_size 512 \
    --output_size 512 \
    --epoch 200001 \



# pretrained_model/adult/stylegan2-ffhq-config-f.pt
# pretrained_model/adult/ffhq_tf_ada.pt
# pretrained_model/anime/stylegan_ada_1024_white_dan20.pt