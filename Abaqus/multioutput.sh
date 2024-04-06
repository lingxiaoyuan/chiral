for folder in arm0 arm1 arm2 arm3 
    do
    cd ${folder}_samples
    qsub ../get_output.sh 0 5
    cd ..
    done
