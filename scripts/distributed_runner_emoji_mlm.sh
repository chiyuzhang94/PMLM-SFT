#!/bin/bash
export PROCID=$OMPI_COMM_WORLD_RANK
export JOB_NUM_NODES=$OMPI_MCA_orte_num_nodes
export LOCALID=$OMPI_COMM_WORLD_LOCAL_RANK
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG_SUBSYS=ALL
/bin/hostname -s
echo "Num of GPUs per nodes, $NPROC_PER_NODE"
echo "Num of Nodes: $JOB_NUM_NODES"
echo "PROCID: $PROCID"
echo "WORLD_RANK(Should be PROCID): $OMPI_COMM_WORLD_RANK"
echo "LOCALID: $LOCALID"
echo "PARENT: $PARENT"
echo "MPORT: $MPORT"
echo "Python: $PYTHONPATH"


python3 -m torch.distributed.launch \
	--nproc_per_node=$NPROC_PER_NODE \
	--nnodes=$JOB_NUM_NODES \
	--node_rank=$PROCID \
	--master_addr="$PARENT" --master_port="$MPORT" \
	$WORK_DIR/language_modeling_emohash_h5.py \
	--gradient_accumulation_steps 4 \
	--train_data_file $WORK_DIR/data/$INPUT_H5 \
	--output_dir $WORK_DIR/$OUT_DIR/ \
	--model_type roberta \
	--logging_dir $WORK_DIR/runs/ \
	--mlm \
	--fp16 \
	--num_workers 0 \
	--warmup_steps 0 \
	--lazy_loading \
	--emoji_mask \
	--model_name_or_path $WORK_DIR/$MODEL_DIR/ \
	--config_name $WORK_DIR/roberta \
	--tokenizer_name $WORK_DIR/roberta \
	--do_train \
	--block_size 64 \
	--learning_rate 5e-5 \
	--num_train_epochs 5 \
	--save_total_limit 5 \
	--per_gpu_train_batch_size 256 \
	--seed 42
