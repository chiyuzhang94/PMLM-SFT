#!/bin/bash

export NPROC_PER_NODE=4
export NCCL_DEBUG=INFO
export HDF5_USE_FILE_LOCKING='FALSE'
export PARENT=`/bin/hostname -s`
export MPORT=13000
export WORK_DIR="PMLM_SFT/"
export HOSTLIST=`cat $PBS_NODEFILE | uniq`
export WORLD_SIZE=`cat $PBS_NODEFILE | wc -l`
export NCCL_SOCKET_NTHREADS=24
export INPUT_H5='emoji_end.h5'
export OUT_DIR='model/'
export MODEL_DIR='model/'

cd $PBS_O_WORKDIR

echo "Python $PYTHONPATH"

mpiexec -x $LD_LIBRARY_PATH $WORK_DIR/scripts/distributed_runner_hashtag_mlm.sh
