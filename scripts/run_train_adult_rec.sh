DEVICES=$1
SEED=6002
BATCH=6
SAVE='./ckpts/ADULT_REC/ckpts/L11'
LOGS='./ckpts/ADULT_REC/logs/L11'
EDIT_CONFIG='./edit_configs/adult/basic_rec.yaml'
E4E_MODEL='./pretrained_model/adult/e4e_ffhq_encode.pt'
CONFIG='./ckpts/ADULT/ckpts/L34/L34_50000.pth'
STYLEGAN='pretrained_model/adult/ffhq_tf_ada.pt'
FINETUNE='./ckpts/ADULT_REC/ckpts/L10/L10_100000.pth'
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
    --finetune $FINETUNE \
