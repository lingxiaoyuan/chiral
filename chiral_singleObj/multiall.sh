cd singleObjective
for pfolder in xx0xx1 xx1xx0 xy0xy1 xy1xy0 yx0yx1 yx1yx0 yy0yy1 yy1yy0 
    do
    cd $pfolder
    for folder in arm0 arm1 arm2 arm3
      do
      cd $folder
        for i in {1..200..1}
          do
          if test -f a${i}.dat;then
          echo ${pfolder} ${folder}
          qsub ../../../all_in_one.sh $i $((i+1))
          fi
          done
      cd ..
      done
    cd ..
    done
cd ..
