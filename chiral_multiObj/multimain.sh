# mode: 'initial',"collect_data","sampling","train", "select","next_simulation", "next"
mode=next
it=1
#xx0xx1xy1yx0 xx1xx0yx1xy1
echo $mode
for obj in xy0xy1xy0yx1 yy0yy1yx0xy1
do
  qsub subjob_main_cpu.sh $obj $mode $it
done
