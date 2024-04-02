
for pfolder in xy0yx0 xy0yx1 xy1yx0 xy1yx1
    do
    cd $pfolder
    for folder in arm0 arm1 arm2 arm3
      do
      cd $folder
        for i in {100..120..1}
          do
          rm -rf $i
          done
      cd ..
      done
    cd ..
    done


