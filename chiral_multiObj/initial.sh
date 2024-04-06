
# mode: 'initial',"collect_data","sampling","train", "select","next_simulation"

for obj in xx0xx1 xx1xx0 xy0xy1 xy1xy0 yx0yx1 yx1yx0 yy0yy1 yy1yy0 xy0yx0 xy0yx1 xy1yx0 xy1yx1 yx0xy0 yx1xy0 yx0xy1 yx1xy1
do
  mkdir $obj 
  cp -r xyyx_initial/** $obj 
done
