cd multiObjective
cd $1
for folder in arm0 arm1 arm2 arm3
    do
    cd $folder
    qsub ../../../write_input_${folder}.sh 0 200
    cd ..
    done
cd ..
done
    
