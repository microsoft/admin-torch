#!/bin/bash
DATADIR=${1:-"./wmt14_en_de_joined_dict"}
MODELDIR=${2:-"None"}

SAVEDIR=${3:-"None"}
UPPER_BOUND=${4:-100}
CP_POINT_NUM=${5:-10}

if [[ $MODELDIR == "None" ]]
then
    if [[ $SAVEDIR == "None" ]]
    then
        echo "SAVEDIR and MODELDIR cannot be None at the same time."
        exit
    fi
    MODELDIR=$SAVEDIR/model_${UPPER_BOUND}_${CP_POINT_NUM}.pt
    if [ -f $MODELDIR  ]; then
        echo $MODELDIR "already exists"
    else
        echo "Start averaging model"
        python average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints ${CP_POINT_NUM}  --output $MODELDIR --checkpoint-upper-bound $UPPER_BOUND | grep 'Finish'
        echo "End averaging model"
    fi
fi

echo "Model path" $MODELDIR

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATADIR \
                    --path $MODELDIR \
                    --batch-size 128 --beam 4 --lenpen 0.6 --remove-bpe \
                    --quiet --fp16
