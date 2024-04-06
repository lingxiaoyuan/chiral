import ml_collections
import torch
import os

def get_default_config():
    config = ml_collections.ConfigDict()
    #workpath
    config.workdir = '/projectnb/twodtransport/lxyuan/chiralFinal/chiral_singleObj/singleObjective'
    config.initial_data_path = '/projectnb/lejlab2/lxyuan/ABAQUSF/chiral/'
    config.design_degree = 7
    config.arm_names = ['arm0', 'arm1', 'arm2', 'arm3']
    config.Sindex = {'xx0':0,'xy0':1,'yx0':2,'yy0':3,'xx1':4,'xy1':5,'yx1':6,'yy1':7}
    config.threshold = 0.03
    config.n_initial = 5
    config.n_next = 10
    config.n_nextpool = 100000
    config.acquistion =  'EI'
    config.xi = 0.002

    #path
    config.path_data= 'data'
    config.path_population= os.path.join(config.workdir, 'population')

    #arm0 configuration
    config.arm0 = arm0 = ml_collections.ConfigDict()
    arm0.arm_name = 'arm0'
    #arm0.feature_name = ['R','theta0','theta1','theta00','theta11','x1','y1','x2','y2','x3','y3']
    #arm0.low = [3,-90,-90,90,-90,-10,-7,-10,-7,-10,-7]
    #arm0.high = [7,-20,-20,160,-20,10,7,10,7,10,7]
    arm0.lig_degree = 7
    arm0.ds = ['poly']*7
    arm0.n_initial = config.n_initial
    arm0.n_nextpool = config.n_nextpool
   
    #config.feature_mean = 0.10607133911057369 
    #config.feature_std = 3.56264247967692
    #arm1 configuration
    config.arm1 = arm1 = ml_collections.ConfigDict()
    arm1.arm_name = 'arm1'
    #arm1.feature_name = ['R','theta0','theta1','theta00','theta11','r_s','x1','y1','x2','y2','x3','y3']
    #arm1.low = [3,-90,-90,90,-90,-10,-7,-10,-7,-10,-7,0.1]
    #arm1.high = [7,-20,-20,160,-20,10,7,10,7,10,7,0.7]
    arm1.lig_degree = 7
    arm1.ds = ['poly']*7
    arm1.n_initial = config.n_initial
    arm1.n_nextpool = config.n_nextpool


    #arm2 configuration
    config.arm2 = arm2 = ml_collections.ConfigDict()
    arm2.arm_name = 'arm2'
    #arm2.feature_name = ['R','theta0','theta1','theta00','theta11','x1','y1','x2','y2','x3','y3']
    #arm2.low = [3,-90,-90,90,-90,-10,-7,-10,-7,-10,-7]
    #arm2.high = [7,-20,-20,160,-20,10,7,10,7,10,7]
    arm2.lig_degree = 7
    arm2.ds = ['poly']*2 + ['sin']*2 + ['poly']*3
    arm2.n_initial = config.n_initial
    arm2.n_nextpool = config.n_nextpool

    #arm3 configuration
    config.arm3 = arm3 = ml_collections.ConfigDict()
    arm3.arm_name = 'arm3'
    #arm3.feature_name = ['R','theta0','theta1','theta00','theta11','r_s','x1','y1','x2','y2','x3','y3']
    #arm3.low = [3,-90,-90,90,-90,-10,-7,-10,-7,-10,-7,0.1]
    #arm3.high = [7,-20,-20,160,-20,10,7,10,7,10,7,0.7]
    arm3.lig_degree = 7
    arm3.ds = ['poly']*7
    arm3.n_initial = config.n_initial
    arm3.n_nextpool = config.n_nextpool


    config.arm_configs = {'arm0':arm0, 'arm1':arm1, 'arm2':arm2, 'arm3':arm3}

    #regression model
    config.reg = reg = ml_collections.ConfigDict()
    reg.workdir = config.workdir
    reg.Sindex = config.Sindex
    reg.model = "regression"
    reg.batch_size = 16
    reg.nepochs = 250
    reg.lr = 0.001
    reg.path_data = 'data'
    reg.input_dim= 208
    reg.output_dim= 1
    reg.aug = False
    reg.name = 'model'
    #reg.seed = 42
    reg.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    reg.n_seed = 15
    #reg.penalty_weight = 10

    return config
