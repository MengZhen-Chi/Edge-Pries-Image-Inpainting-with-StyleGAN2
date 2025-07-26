DEVICES=$1
SEED=6002
BATCH=6
SAVE='./ckpts/CAT_REC/ckpts/L4'
LOGS='./ckpts/CAT_REC/logs/L4'
EDIT_CONFIG='./edit_configs/cat/basic_rec.yaml'
E4E_MODEL='./ckpts/alchemy_v3/1230_models/cat_models/best_model.pt'
CONFIG='./ckpts/alchemy_v3/1230_models/cat_models/cat_editor_v3_1230.pth'
STYLEGAN='./ckpts/alchemy_v3/1230_models/cat_models/cat_decoder_v3_1230.pt'
FINETUNE='./ckpts/CAT_REC/ckpts/L3/L3_50000.pth'
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
    --finetune $FINETUNE \
