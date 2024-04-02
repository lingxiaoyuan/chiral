
# mode: 'initial',"collect_data","sampling","train", "select","next_simulation", "next"
mode=next
it=$1

echo $mode
echo $it
#xx0xx1 xx1xx0 xy0xy1 xy1xy0 yx0yx1 yx1yx0 yy0yy1 yy1yy0 
for obj in xy0yx0 xy0yx1 xy1yx0 xy1yx1 yx0xy0 yx1xy0 yx0xy1 yx1xy1
do
  qsub subjob_main_cpu.sh $obj $mode $it
done
