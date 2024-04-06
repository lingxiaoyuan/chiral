#!/bin/bash -l
#$ -P twodtransport
#$ -N inputINPfile
#$ -l h_rt=12:00:00
#$ -j y
#$ -pe omp 8
source /ad/eng/bin/engenv.sh
module load simulia/2020
abaqus cae noGUI=/projectnb/lejlab2/lxyuan/ABAQUSF/chiral/scripts/main_1step_arm1.py -- $1 $2
#change the above path to where you save /Abaqus/scripts/main_1step_arm1.py