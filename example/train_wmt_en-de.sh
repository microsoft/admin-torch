DATA_PATH=${1:-"./wmt14_en_de_joined_dict"}
LAYERS=${2:-18}
OUTPUT_PATH=${3:-"./admin_18L_asParameter"}
ADDITIONAL_ARGS=${4:-"--share-all-embeddings --encoder-as-parameter --decoder-as-parameter"}

TOKENS=4096
DEVICE_NUMBER=8
FREQ=1

NUMBER_OF_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ $NUMBER_OF_GPUS != $DEVICE_NUMBER ]]
then
    echo "The script is for $DEVICE_NUMBER card, but only find $NUMBER_OF_GPUS cards."
    echo "Please modify TOKENS, DEVICES, and FREQ in the script accordingly."
    echo 
    echo "Note that you need to keep device_number * tokens * freq = 32768"
    exit    
fi

DEVICE_LIST=$(( DEVICE_NUMBER - 1 ))
DEVICE_LIST=$(seq -s "," 0 $DEVICE_LIST)

echo "Using GPUs $DEVICE_LIST for training"

CUDA_VISIBLE_DEVICES=$DEVICE_LIST fairseq-train \
    $DATA_PATH $ADDITIONAL_ARGS -s en -t de \
    --arch transformer_wmt_en_de \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --max-update 1000000 \
    --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.1 \
    --max-tokens $TOKENS --update-freq $FREQ \
    --save-dir $OUTPUT_PATH --seed 1111 --restore-file x.pt \
    --log-format simple --log-interval 30 --memory-efficient-fp16 \
    --encoder-layers $LAYERS --decoder-layers $LAYERS \
    --threshold-loss-scale 0.0625 --fp16-scale-window 256 --fp16 

