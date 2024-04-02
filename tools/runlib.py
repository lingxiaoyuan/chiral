import numpy as np
import random
import collections
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torchvision.transforms as transforms
import logging
import argparse
import os, sys
from copy import deepcopy
from networks.NNs import *
from tools.helpers import * 
import functools
import matplotlib.pyplot as plt

from scipy.stats import norm

class mse_error():
    def __init__(self,config):
        self.device = config.device

    def __call__(self,model,dataset_loader):
        error = 0
        b = 0
        criterion = nn.MSELoss(reduction = "mean")
        for x, y in dataset_loader:
            b+=1
            x = x.to(self.device)
            predicted = model(x).detach().cpu()
            error += criterion(predicted,y)
        return error/b

def get_data(config, datapath, seed= 42):
    batch_size = config.batch_size
    data = np.loadtxt(datapath)
    x = data[:,:config.input_dim]
    y = data[:,-config.output_dim:]  
    x_train, x_val, y_train,y_val = train_test_split(x,y, test_size = 0.25, shuffle = True, random_state = seed)

    x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
    x_val, y_val = torch.Tensor(x_val), torch.Tensor(y_val)
    if config.model == "classification":
        y_train, y_val = y_train.long(), y_val.long()

    train_set = TensorDataset(x_train,y_train)
    val_set = TensorDataset(x_val,y_val)

    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = False)
    val_dataloader = DataLoader(val_set, batch_size = batch_size, shuffle = False)

    return train_dataloader, val_dataloader

def predict_y(config, model, dataloader):
    model = model.to(config.device)
    model.eval()
    y_hat = torch.zeros([0,config.output_dim]).to(config.device)
    #don't forget the comma here
    for x_batch, in dataloader:
        x_batch = x_batch.to(config.device)
        output = model(x_batch)
        y_hat = torch.concat((y_hat,output),0)
    return y_hat.cpu().detach().numpy()

def evaluate_y(config, model, dataloader):
    model = model.to(config.device)
    model.eval()
    y_hat = np.zeros([0,config.output_dim])
    y_gt = np.zeros([0,config.output_dim])
    for x_batch,y_batch in dataloader:
        x_batch = x_batch.to(config.device)
        output = model(x_batch).cpu().detach().numpy()
        y_hat = np.concatenate((y_hat,output),0)
        y_gt = np.concatenate((y_gt,y_batch),0)
    return y_gt, y_hat

def train(objective, config, it, seed=42):
    path_log  = os.path.join(config.workdir,objective, "./logs/")
    logger = get_logger(os.path.join(path_log,'training_history.log'),display=False, save = True)
    logger.info("iteration: {}".format(it))
    logger.info("seed: {}".format(seed))

    obejective_path =  os.path.join(config.workdir, objective)
    path_model = os.path.join(obejective_path,'saved_models')
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    datapath = os.path.join(obejective_path, config.path_data, "objective_iter%s.dat"%it)
    train_dataloader, val_dataloader = get_data(config, datapath, seed= seed)
    model = NNs_reg(config.input_dim, config.output_dim).to(config.device)
    loss_func = nn.MSELoss(reduction = "mean")
    evaluation = mse_error(config)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    num_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(model)
    logger.info("Number of parameters: {}".format(num_para))

    min_err = float("inf")
    plt.figure()
    train_err_h, val_err_h = [], []
    for epoch in range(config.nepochs):
        for x_batch,y_batch in train_dataloader:
            optimizer.zero_grad()
            model.train()
            x_batch = x_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            output = model(x_batch)
            loss = loss_func(output,y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_err = evaluation(model,val_dataloader)
            train_err = evaluation(model,train_dataloader)
            logger.info("Epoch {:04d} | Train err {:.4f} | Val err {:.4f}".format(epoch, train_err, val_err))
            train_err_h.append(train_err)
            val_err_h.append(val_err)
            ##save model
            if val_err < min_err:
                torch.save({'state_dict': model.state_dict(), 'config': config}, os.path.join(path_model, config.name+ '_iter%s_seed%s.pth'%(it,seed)))
                min_err = val_err
    
    #results
    savepath = os.path.join(obejective_path,'results')
    if not os.path.exists(savepath): os.mkdir(savepath)


    #training history
    plt.plot(range(config.nepochs), train_err_h,  c = 'k', label = 'train err')
    plt.plot(range(config.nepochs), val_err_h, c = 'r', label = 'val err')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(os.path.join(savepath,'training_iter%s_seed%s.png'%(it,seed)),bbox_inches='tight')

    #performance
    ytrain_gt, ytrain_hat = evaluate_y(config, model, train_dataloader)
    ytest_gt, ytest_hat = evaluate_y(config, model, val_dataloader)
    
    plt.figure(figsize= (8,5))
    for i in range(config.output_dim):
        plt.scatter(ytrain_gt[:,i],ytrain_hat[:,i], s=12,marker = 'o',label = 'train r2 = %f'%r2_score(ytrain_gt[:,i],ytrain_hat[:,i]))
        plt.scatter(ytest_gt[:,i],ytest_hat[:,i], s=12, marker = 'o',label = 'val r2 = %f'%r2_score(ytest_gt[:,i],ytest_hat[:,i]))
        plt.plot(ytrain_gt[:,i],ytrain_gt[:,i])
    #plt.legend(['kxx-','kxy-','kyx-','kyy-','kxx+','kxy+','kyx+','kyy+'])
    lgd = plt.legend(loc=(1.0,0))
    plt.xlabel('Groud Truth')
    plt.ylabel('Prediction')
    #plt.tight_layout()
    plt.savefig(os.path.join(savepath,'performance_iter%s_seed%s.png'%(it,seed)), bbox_extra_artists=(lgd,), bbox_inches='tight')

    #save predicted properties for training data
    x_train = np.loadtxt(datapath)[:,:config.input_dim]
    x_train = TensorDataset(torch.Tensor(x_train))
    x_train_loader = DataLoader(x_train,batch_size = config.batch_size, shuffle = False)
    ytrain_hat = predict_y(config, model, x_train_loader)
    np.savetxt(os.path.join(savepath,'muObserved_iter%s_seed%s.dat'%(it,seed)), ytrain_hat)
    
def trainEnsemble(objective, config,it):
    for seed in range(config.n_seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        train(objective, config, it, seed)

def expected_improvement(mu,sigma, mu_observed,xi=0.01):
    mu_observed_opt = np.max(mu_observed)
    with np.errstate(divide='warn'):
        res= mu-mu_observed_opt-xi
        Z = res/sigma
        EI = res*norm.cdf(Z) + sigma*norm.pdf(Z)
        EI[sigma==0.0] = 0.0
    return EI.ravel()

## The function is_pareto_efficient is copied from https://github.com/QUVA-Lab/artemis/blob/peter/artemis/general/pareto_efficiency.py
## with minor change such that the pareto front are from maximizing objective rather than minimizing objective
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def acquistion_single(objective, config,it):
    path_log  = os.path.join(config.workdir,objective, "./logs/")
    logger = get_logger(os.path.join(path_log,'optimization_history.log'),display=False, save = True)
    logger.info('obejective: {}'.format(objective))
    logger.info('iteration: {}'.format(it))
    logger.info('seed number: {}'.format(config.reg.n_seed))
    if it == 0:
        arm_index_unlabel_all = {it:{}}
        for arm_name in config.arm_names:
            arm_config = config.arm_configs[arm_name]
            arm_index_unlabel_all[it][arm_name] = np.array(range(config.n_nextpool))
        f = open(os.path.join(config.workdir,objective,config.path_data, 'index_unlabeled.pkl'),'wb')
        pickle.dump(arm_index_unlabel_all, f)
        f.close

    else:
        f = open(os.path.join(config.workdir,objective,config.path_data, 'index_unlabeled.pkl'),'rb')
        arm_index_unlabel_all = pickle.load(f)
    
    arm_index_unlabel  =  deepcopy(arm_index_unlabel_all[it])

    y_seed = []
    y_observed = []
    arm_select = {}
    y_hat_all = []
    datapath = os.path.join(config.workdir, objective,'results')
    r = [config.Sindex[objective[:3]], config.Sindex[objective[3:]]]
    for seed in range(config.reg.n_seed):  
        y_arm = []
        #get the prediction for unlabelled test dat in each arm and concate them together
        for arm_name in config.arm_names:
            x_test = np.loadtxt(os.path.join(config.path_population,'%s.dat'%arm_name))[arm_index_unlabel[arm_name]]
            x_test = TensorDataset(torch.Tensor(x_test))
            x_test_loader = DataLoader(x_test,batch_size = config.reg.batch_size, shuffle = False)
            model = NNs_reg(config.reg.input_dim, config.reg.output_dim)         
            model = model.to(config.reg.device)
            path_model_trained = os.path.join(config.workdir,objective, 'saved_models', config.reg.name+ '_iter%s_seed%s.pth'%(it,seed))
            model.load_state_dict(torch.load(path_model_trained,map_location=config.reg.device)['state_dict'])
            y_hat = predict_y(config.reg, model, x_test_loader)
            y_hat_all.append(y_hat)
        #the length of y_arm is the sum of unlabeled data length
        y_observed.append(np.loadtxt(os.path.join(datapath,'muObserved_iter%s_seed%s.dat'%(it,seed))))
    y_ratio = np.concatenate(y_hat_all).reshape(config.reg.n_seed,-1)
    mu = y_ratio.mean(axis=0)
    sigma = y_ratio.std(axis=0)
    mu_observed = np.array(y_observed).mean(axis=0)
    logger.info('mu_predict_y : {}'.format(mu.mean()))
    logger.info('sigma_predict_y : {}'.format(sigma.mean()))
    
    if config.acquistion == 'UCB':
        EI = mu + config.beta*sigma
    else:
        EI = expected_improvement(mu,sigma, mu_observed, xi=config.xi)
    index_max = np.sort(np.argpartition(EI, -config.n_next)[-config.n_next:])
    logger.info('selected index in unlabeled data {}'.format(index_max))
    logger.info('predicted objective {}'.format(np.exp(mu[index_max])))
    logger.info('predicted std of objective {}'.format(np.exp(sigma[index_max])))

    #get the selected data index of each arm
    curr_armlen =  0
    for arm_name in config.arm_names:
        length_unlabel = arm_index_unlabel[arm_name].shape[0]
        select_index = np.intersect1d(index_max[index_max >= curr_armlen], index_max[index_max < length_unlabel + curr_armlen]) - curr_armlen
        arm_select[arm_name] = arm_index_unlabel[arm_name][select_index]
        #remove the data to be labeled
        arm_index_unlabel[arm_name] = np.delete(arm_index_unlabel[arm_name], select_index)
        #move the pointer to the first index of the next arm
        curr_armlen+=length_unlabel
    logger.info('selected data index in each arm {}'.format(arm_select))
    logger.info('current length of each arm {}'.format([arm_index_unlabel[arm_name].shape[0] for arm_name in config.arm_names]))

    #save the unlabeled data index for the next iteration
    arm_index_unlabel_all[it+1] = arm_index_unlabel
    f = open(os.path.join(config.workdir,objective,config.path_data, 'index_unlabeled.pkl'),'wb')
    pickle.dump(arm_index_unlabel_all, f)
    f.close

    #save the index of the selected data 
    if it==0:
        arm_index_label_all = {}
    else:
        f = open(os.path.join(config.workdir,objective,config.path_data, 'index_labeled.pkl'),'rb')
        arm_index_label_all = pickle.load(f)

    arm_index_label_all[it+1] = arm_select
    f = open(os.path.join(config.workdir,objective,config.path_data, 'index_labeled.pkl'),'wb')
    pickle.dump(arm_index_label_all, f)
    f.close

def acquistion_multi(objective, config,it):
    path_log  = os.path.join(config.workdir,objective, "./logs/")
    logger = get_logger(os.path.join(path_log,'optimization_history.log'),display=False, save = True)
    logger.info('obejective: {}'.format(objective))
    logger.info('iteration: {}'.format(it))
    logger.info('seed number: {}'.format(config.reg.n_seed))
    if it == 0:
        arm_index_unlabel_all = {it:{}}
        for arm_name in config.arm_names:
            arm_config = config.arm_configs[arm_name]
            arm_index_unlabel_all[it][arm_name] = np.array(range(config.n_nextpool))
        f = open(os.path.join(config.workdir,objective,config.path_data, 'index_unlabeled.pkl'),'wb')
        pickle.dump(arm_index_unlabel_all, f)
        f.close

    else:
        f = open(os.path.join(config.workdir,objective,config.path_data, 'index_unlabeled.pkl'),'rb')
        arm_index_unlabel_all = pickle.load(f)
    
    arm_index_unlabel  =  deepcopy(arm_index_unlabel_all[it])

    y_seed = []
    y_observed = []
    arm_select = {}
    y_hat_all = []
    datapath = os.path.join(config.workdir, objective,'results')
    for seed in range(config.reg.n_seed):  
        y_arm = []
        #get the prediction for unlabelled test dat in each arm and concate them together
        for arm_name in config.arm_names:
            x_test = np.loadtxt(os.path.join(config.path_population,'%s.dat'%arm_name))[arm_index_unlabel[arm_name]]
            x_test = TensorDataset(torch.Tensor(x_test))
            x_test_loader = DataLoader(x_test,batch_size = config.reg.batch_size, shuffle = False)
            model = NNs_reg(config.reg.input_dim, config.reg.output_dim)         
            model = model.to(config.reg.device)
            path_model_trained = os.path.join(config.workdir,objective, 'saved_models', config.reg.name+ '_iter%s_seed%s.pth'%(it,seed))
            model.load_state_dict(torch.load(path_model_trained,map_location=config.reg.device)['state_dict'])
            y_hat = predict_y(config.reg, model, x_test_loader)
            y_hat_all.append(y_hat)
        #the length of y_arm is the sum of unlabeled data length
        y_observed.append(np.loadtxt(os.path.join(datapath,'muObserved_iter%s_seed%s.dat'%(it,seed))))
    y_ratio = np.concatenate(y_hat_all).reshape(config.reg.n_seed,-1,config.reg.output_dim)
    y_ratio = np.array(y_ratio,dtype="float64")
    mu = y_ratio.mean(axis=0)
    sigma = y_ratio.std(axis=0)
    mu_observed = np.array(y_observed).mean(axis=0)
    logger.info('mu_predict_y : {}'.format(mu.mean()))
    logger.info('sigma_predict_y : {}'.format(sigma.mean()))
    
    if config.acquistion == 'UCB':
        #dimension of EI: (n_data,n_objective)
        EI = mu + config.beta*sigma
    else:
        EI = expected_improvement(mu,sigma, mu_observed, xi=config.xi)
    pareto_index = is_pareto_efficient(EI, return_mask=False)
    optimal_datapoints=EI[pareto_index]
    index_max = np.array(pareto_index)[(optimal_datapoints>1).all(axis = 1)]
    #index_max = np.sort(np.argpartition(EI, -config.n_next)[-config.n_next:])
    logger.info('selected index in unlabeled data {}'.format(index_max))
    if len(index_max)==0:
        logger.info('No eligible pareto front')
        return False
    #logger.info('predicted objective {}'.format(np.exp(mu[index_max])))
    #logger.info('predicted std of objective {}'.format(np.exp(sigma[index_max])))

    #get the selected data index of each arm
    curr_armlen =  0
    for arm_name in config.arm_names:
        length_unlabel = arm_index_unlabel[arm_name].shape[0]
        select_index = np.intersect1d(index_max[index_max >= curr_armlen], index_max[index_max < length_unlabel + curr_armlen]) - curr_armlen
        arm_select[arm_name] = arm_index_unlabel[arm_name][select_index]
        #remove the data to be labeled
        arm_index_unlabel[arm_name] = np.delete(arm_index_unlabel[arm_name], select_index)
        #move the pointer to the first index of the next arm
        curr_armlen+=length_unlabel
    logger.info('selected data index in each arm {}'.format(arm_select))
    logger.info('current length of each arm {}'.format([arm_index_unlabel[arm_name].shape[0] for arm_name in config.arm_names]))

    #save the unlabeled data index for the next iteration
    arm_index_unlabel_all[it+1] = arm_index_unlabel
    f = open(os.path.join(config.workdir,objective,config.path_data, 'index_unlabeled.pkl'),'wb')
    pickle.dump(arm_index_unlabel_all, f)
    f.close

    #save the index of the selected data 
    if it==0:
        arm_index_label_all = {}
    else:
        f = open(os.path.join(config.workdir,objective,config.path_data, 'index_labeled.pkl'),'rb')
        arm_index_label_all = pickle.load(f)

    arm_index_label_all[it+1] = arm_select
    f = open(os.path.join(config.workdir,objective,config.path_data, 'index_labeled.pkl'),'wb')
    pickle.dump(arm_index_label_all, f)
    f.close
    return True

