from threading import Timer
import subprocess
import time
import os

'''
def subjob_multimain(it):
    subprocess.run(["sh", "multimain.sh", it])

def subjob_multiall():
    subprocess.run(["sh", "multiall.sh"])

'''

#training model
workpath = "singleObjective"
def subjob_multimain(obj,it):
    mode = "next"
    subprocess.run(["qsub", "subjob_main_cpu.sh", obj, mode, str(it)])

#run FEM simulation
def subjob_multiall(obj):
    for folder in ["arm0", "arm1", "arm2", "arm3"]:
        path = os.path.join(workpath, obj,folder)
        for i in range(200):
            if os.path.exists(os.path.join(path,'a%s.dat'%i)):
                subprocess.check_output(["qsub", "../../../all_in_one.sh", str(i), str(i+1)], cwd=path)


objectives_all = ['xy0yx0', 'xy0yx1', 'xy1yx0', 'xy1yx1', 'yx0xy0', 'yx1xy0', 'yx0xy1', 'yx1xy1']
#objectives_all = ["xx0xx1",  "xx1xx0", "xy0xy1", "xy1xy0", "yx0yx1", "yx1yx0", "yy0yy1", "yy1yy0"]
for it in range(0,10):
    print('training',it)
    for obj in objectives_all: 
        subjob_multimain(obj,it)
    if it<4: time.sleep(4800)
    else: time.sleep(7200)

    print('simulation',it)
    for obj in objectives_all: 
        subjob_multiall(obj)
    time.sleep(1800)
    for obj in objectives_all: 
        subjob_multiall(obj)
    time.sleep(1800)
