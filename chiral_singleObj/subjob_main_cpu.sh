#!/bin/bash -l
#quest 1 core. This will set NSLOTS=1
#$ -pe omp 8
# Terminate after 12 hours
#$ -l h_rt=4:00:00

# Join output and error streams
#$ -j y
# Specify Project
#$ -P twodtransport
# Give the job a name
#$ -N zjobhistory

# load modules:
module load python3/3.8.10
module load pytorch/1.12.1
export PYTHONPATH=/projectnb/twodtransport/lxyuan/packages/lib/python3.8/site-packages/:$PYTHONPATH

echo "main:"
python main.py \
  --objective=$1 \
  --mode=$2 \
  --it=$3
