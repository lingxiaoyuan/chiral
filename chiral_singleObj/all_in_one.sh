#!/bin/bash -l
#$ -P twodtransport
#$ -N simulation
#$ -l h_rt=2:00:00
#$ -j y
#$ -pe omp 8
source /ad/eng/bin/engenv.sh
module load simulia/2020

for ((i=$1;i<$2;i++))
do
 if test -f a${i}.dat;then
    mkdir $i
    cd $i
    mv ../a${i}.dat a.dat
    mv ../circle${i}.dat circle.dat
    mv ../point${i}.dat point.dat
    for j in {0..3}
    do
    name1='one_ligament_test'
    filename=${name1}${j}
    mv ../disp${i}_test${j}.dat disp_${j}.dat
    mv ../a${i}_${filename}.inp ${filename}.inp
    abaqus input=./${filename}.inp job=${filename} cpus=$NSLOTS interactive
    done
    cd ..
 fi
done
abaqus cae noGUI=/projectnb/lejlab2/lxyuan/ABAQUSF/chiral/scripts/getdata_1step.py -- $1 $2
#change the above path to where you save /Abaqus/scripts/getdata_1step.py
