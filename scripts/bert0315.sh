#!/bin/bash

GPUS_PER_NODE=4
MASTER_ADDR=`head -n 1 $SGE_JOB_HOSTLIST`
MASTER_PORT=8888

mpiexec -N $GPUS_PER_NODE \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    python pretrain_bert.py \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 16 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save checkpoints/$JOB_NAME \
       --load checkpoints/$JOB_NAME \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --shuffle \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-layernorm \
       --fp32-embedding
