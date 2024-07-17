NUM_GPUS=8
WORK_PATH=$(pwd)
TRAIN_PATH=${WORK_PATH}/run_llms.py 
SIGNATURE=patch_train_pre
CHECKPOINT_PATH=${WORK_PATH}/${SIGNATURE}
mkdir ${CHECKPOINT_PATH}
LOG_FILE=${CHECKPOINT_PATH}/log.$(date +%s)

TOKENIZER_PATH=${WORK_PATH}/llama2_tokenizer
DATASET=""
for i in $(seq -w 0 19); do
    if [[ $i -eq 0 ]]; then
        DATASET=${WORK_PATH}/pile_uncopyrighted/${i}.text.jsonl
    else
        DATASET=${DATASET},${WORK_PATH}/pile_uncopyrighted/${i}.text.jsonl
    fi
done
DATASET_VALID=${WORK_PATH}/wikitext_document_level-test.json
export PYTHONPATH=~/mylibs:$PYTHONPATH

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth,en,em,bond
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export CXX=g++
export MASTER_ADDR="${CHIEF_IP:=localhost}"
MASTER_PORT=$((1 + $RANDOM % 99999))
export TRITON_CACHE_DIR=$WORK_PATH/cache/
export TRANSFORMERS_CACHE=${WORK_PATH}/cache/
export HF_HOME=${WORK_PATH}/cache/
export TORCH_EXTENSIONS_DIR=${WORK_PATH}/cache/torch_extension/${model_name}
export OMP_NUM_THREADS=20
TOKENIZERS_PARALLELISM=false
HOST_NUM=1
INDEX=0

torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node $NUM_GPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${TRAIN_PATH} \
    --model_type llama \
    --tokenizer_name $TOKENIZER_PATH \
    --patch_size 4 \
    --config_overrides "hidden_size=1024,intermediate_size=2752,num_hidden_layers=24,num_attention_heads=16,num_key_value_heads=16" \
    --train_file $DATASET \
    --validation_file $DATASET_VALID \
    --keep_linebreaks True \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --block_size 8192 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --streaming \
    --seed 1 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --max_steps 90000 \
    --save_strategy "steps" \
    --save_steps 15000 \
    --save_total_limit 50 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --learning_rate 3e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --do_train \
    --do_eval \
    --ddp_timeout 3600 \
    --output_dir $CHECKPOINT_PATH \
    --cache_dir $TRANSFORMERS_CACHE \
    --overwrite_output_dir \
    --overwrite_cache \
    --bf16 True \
    2>&1 |tee ${LOG_FILE}

DATASET=""
for i in $(seq -w 20 29); do
    if [[ $i -eq 20 ]]; then
        DATASET=${WORK_PATH}/pile_uncopyrighted/${i}.text.jsonl
    else
        DATASET=${DATASET},${WORK_PATH}/pile_uncopyrighted/${i}.text.jsonl
    fi
done
SIGNATURE=patch_train_post
CHECKPOINT_PATH=${WORK_PATH}/${SIGNATURE}
mkdir ${CHECKPOINT_PATH}
LOG_FILE=${CHECKPOINT_PATH}/log.$(date +%s)

torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node $NUM_GPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${TRAIN_PATH} \
    --model_type llama \
    --tokenizer_name $TOKENIZER_PATH \
    --patch_size 1 \
    --model_name_or_path ${WORK_PATH}/patch_train_pre \
    --train_file $DATASET \
    --validation_file $DATASET_VALID \
    --keep_linebreaks True \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --block_size 2048 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --streaming \
    --seed 1 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 1 \
    --max_steps 45000 \
    --save_strategy "steps" \
    --save_steps 15000 \
    --save_total_limit 50 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --learning_rate 3e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --do_train \
    --do_eval \
    --ddp_timeout 3600 \
    --output_dir $CHECKPOINT_PATH \
    --cache_dir $TRANSFORMERS_CACHE \
    --overwrite_output_dir \
    --overwrite_cache \
    --bf16 True \
    2>&1 |tee ${LOG_FILE}
