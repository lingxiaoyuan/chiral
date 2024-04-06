#from threading import Timer
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
workpath = "multiObjective"
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


objectives_all = ['xx0xx1xy0yx1', 'xx0xx1xy1yx0', 'xx0xx1xy0yx0','xx1xx0xy1yx0','xx1xx0xy0yx1', 'xx1xx0xy0yx0']
for it in range(11):
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
