cd singleObjective
for pfolder in xx0xx1 xx1xx0 xy0xy1 xy1xy0 yx0yx1 yx1yx0 yy0yy1 yy1yy0 
    do
    cd $pfolder
    for folder in arm0 arm1 arm2 arm3
        do
        cd $folder
        qsub ../../../write_input_${folder}.sh 1 100
        cd ..
        done
    cd ..
    done
    
