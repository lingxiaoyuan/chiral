#!/bin/bash -l
#quest 1 core. This will set NSLOTS=1
#$ -pe omp 4
# Request 1 GPU
#$ -l gpus=1
# Request at least compute capability 3.5, if higer, 6.0
#$ -l gpu_c=6.0
# Terminate after 12 hours
#$ -l h_rt=12:00:00

# Join output and error streams
#$ -j y
# Specify Project
#$ -P twodtransport
# Give the job a name
#$ -N zjobhistory

# load modules:
module load python3/3.8.10
module load pytorch/1.12.1

echo "main:"
python main.py \
  --objective=$1 \
  --mode=$2 \
  --it=$3
