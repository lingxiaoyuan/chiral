cd singleObjective
cd $1
for folder in arm0 arm1 arm2 arm3
    do
    cd $folder
    qsub ../../../write_input_${folder}.sh 1 200
    cd ..
    done
cd ..
done
    
