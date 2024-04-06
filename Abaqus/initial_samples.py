import numpy as np
import pandas as pd
import math
import random
import os,sys
import pickle
from numpy.linalg import lstsq,norm 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


sys.path.append('../')
sys.path.append('../tools')
sys.path.append('../chiral_singleObj')

from tools.helpers import *
from tools.lig_space import *
from tools.data_processing import write_data

from tools.arm0_design import asym_poly_designs as arm0_asym_poly_designs
from tools.arm1_design import asym_poly_designs as arm1_asym_poly_designs
from tools.arm2_design import asym_poly_designs as arm2_asym_poly_designs
from tools.arm3_design import asym_poly_designs as arm3_asym_poly_designs

from configs.configs_default import get_default_config
config = get_default_config()

for arm_name in config.arm_names:
    arm_config = config.arm_configs[arm_name]
    simulation_path = os.path.join(arm_name+'_samples')
    if not os.path.exists(simulation_path): os.mkdir(simulation_path)
    asym_poly_designs = globals()[f'{arm_name}_asym_poly_designs']
    ligs, circles, points = asym_poly_designs(n_designs=5, n = arm_config.lig_degree, ds_name=arm_config.ds)
    write_data(ligs=ligs, circles=circles, points=points, path=simulation_path, start=0)
