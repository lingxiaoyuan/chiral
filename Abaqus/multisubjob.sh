for folder in arm0 arm1 arm2 arm3
  do
  cd ${folder}_samples
    for i in {0..5..2}
      do
      qsub ../subjobs_args.sh $i $((i+2))
      done
  cd ..
  done


