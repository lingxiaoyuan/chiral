#!/bin/bash -l
#$ -P twodtransport
#$ -N output_data
#$ -l h_rt=4:00:00
#$ -j y
#$ -pe omp 4
source /ad/eng/bin/engenv.sh
module load simulia/2020
abaqus cae noGUI=/projectnb/lejlab2/lxyuan/ABAQUSF/chiral/scripts/getdata_1step.py -- $1 $2
#change the above path to where you save /Abaqus/scripts/getdata_1step.py