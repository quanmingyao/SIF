#!/usr/bin/env bash
set -o xtrace

NL=`echo -ne '\015'`

function screen_run {
        local title=$2
        local cmd="$3"

        screen -S $1 -X screen -t $title
        screen -S $1 -p $title -X stuff "$cmd$NL"
}

SCREEN_NAME=AutoMC_ml100k@xiangning
screen -S $SCREEN_NAME -X quit
screen -d -m -S $SCREEN_NAME -t shell -s /bin/bash
sleep 1


screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=ncf --embedding_dim=1 --gpu=6 --weight_decay=5e-5 --process_name=ml-100k_ncf1@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=ncf --embedding_dim=1 --gpu=7 --weight_decay=1e-6 --process_name=ml-100k_ncf2@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=ncf --embedding_dim=1 --gpu=5 --weight_decay=5e-6 --process_name=ml-100k_ncf3@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=ncf --embedding_dim=1 --gpu=6 --weight_decay=1e-5 --process_name=ml-100k_ncf4@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=deepwide --embedding_dim=1 --gpu=7 --weight_decay=5e-5 --process_name=ml-100k_deepwide1@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=deepwide --embedding_dim=1 --gpu=5 --weight_decay=1e-6 --process_name=ml-100k_deepwide2@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=deepwide --embedding_dim=1 --gpu=6 --weight_decay=5e-6 --process_name=ml-100k_deepwide3@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=deepwide --embedding_dim=1 --gpu=7 --weight_decay=1e-5 --process_name=ml-100k_deepwide4@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=plus --embedding_dim=1 --gpu=5 --weight_decay=5e-5 --process_name=ml-100k_plus1@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=plus --embedding_dim=1 --gpu=6 --weight_decay=1e-6 --process_name=ml-100k_plus2@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=plus --embedding_dim=1 --gpu=7 --weight_decay=5e-6 --process_name=ml-100k_plus3@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k --mode=plus --embedding_dim=1 --gpu=5 --weight_decay=1e-5 --process_name=ml-100k_plus4@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k altgrad --embedding_dim=1 --gpu=6 --weight_decay=5e-5 --process_name=ml-100k_altgrad1@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k altgrad --embedding_dim=1 --gpu=7 --weight_decay=1e-6 --process_name=ml-100k_altgrad2@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k altgrad --embedding_dim=1 --gpu=5 --weight_decay=5e-6 --process_name=ml-100k_altgrad3@xiangning --seed=1"

screen_run $SCREEN_NAME c "
python3 ~/AutoMC/main.py --dataset=ml-100k altgrad --embedding_dim=1 --gpu=6 --weight_decay=1e-5 --process_name=ml-100k_altgrad4@xiangning --seed=1"

