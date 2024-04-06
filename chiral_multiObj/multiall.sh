cd multiObjective
for pfolder in xy0xy1xy0yx1 yy0yy1yx0xy1
    do
    cd $pfolder
    for folder in arm0 arm1 arm2 arm3
      do
      cd $folder
        for i in {100..112..12}
          do
          qsub ../../all_in_one_${folder}.sh $i $((i+12))
          done
      cd ..
      done
    cd ..
    done
cd ..