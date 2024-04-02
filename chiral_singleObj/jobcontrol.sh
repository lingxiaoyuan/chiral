#!/bin/bash -l
#quest 1 core. This will set NSLOTS=1
#$ -pe omp 8
# Terminate after 36 hours
#$ -l h_rt=36:00:00

# Join output and error streams
#$ -j y
# Specify Project
#$ -P twodtransport
# Give the job a name
#$ -N Iteration

# load modules:
module load python3/3.8.10

echo "jobControl"
python jobcontrol.py