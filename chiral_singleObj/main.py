import numpy as np
import pandas as pd
import math
import random
import shutil
from absl import app,flags
import ml_collections
from ml_collections.config_flags import config_flags
from configs.configs_default import get_default_config
import logging
import os,sys
import yaml
from pathlib import Path
import subprocess

sys.path.append("../")
sys.path.append("../tools")
from tools.data_processing import *
from tools.runlib import *

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("objective", 'xy0yx0', "Objective to be maximized")
flags.DEFINE_enum("mode", None, ["collect_data","sampling","train", "select","next_simulation","next"], "Running mode")
flags.DEFINE_string("eval_folder", "eval","The folder name for storing evaluation results")
flags.DEFINE_integer("it", None, "number of iteration")
flags.mark_flags_as_required(["mode"])

def main(argv):
    #collect data
    it = FLAGS.it
    objective = FLAGS.objective
    config = get_default_config()
    path_data = os.path.join(config.workdir,objective,config.path_data)
    Path(path_data).mkdir(parents=True, exist_ok=True)
    path_log  = os.path.join(config.workdir,objective, "./logs/")
    Path(path_log).mkdir(parents=True, exist_ok=True)

    #logger = get_logger(os.path.join(path_log,'history.log'), display=False, save = True)

    if FLAGS.mode == 'sampling':
        sampling(config)
        test_features(config)

    elif FLAGS.mode == 'next':
        if it == 0: initial_prepare(objective, config)
        get_train_data(objective, config, it)
        trainEnsemble(objective, config.reg, it)
        acquistion_single(objective,config,it)
        next_simulation(objective,config,it+1) 
        subprocess.run(["sh", "writeinput.sh",objective])
        #np.save('%s%s.dat'%(objective,str(it)),np.ones(1))

    elif FLAGS.mode == "collect_data":
        get_train_data(objective, config, it)
        
    elif FLAGS.mode == 'train':
        trainEnsemble(objective, config.reg, it)
        
    elif FLAGS.mode == 'select':
        acquistion_single(objective,config,it)
    
    elif FLAGS.mode == 'next_simulation':
        next_simulation(objective,config,it) 
        subprocess.run(["sh", "writeinput.sh",objective])


if __name__ == "__main__":
    app.run(main)
