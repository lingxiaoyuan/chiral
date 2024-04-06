for folder in arm0 arm1 arm2 arm3 
    do
    cd ${folder}_samples
    qsub ../write_input_${folder}.sh 0 100
    cd ..
    done
