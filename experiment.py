import argparse
import os
# import subprocess

import train
import test 
from copy import deepcopy

import myconfig
from utils.bdd import BinaryDecisionDiagram
import utils.util as util



import pandas as pd
import numpy  as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

import torch

avg_list = lambda val_s: sum(val_s) / len(val_s)



def prepare_path(args):

    train_filename = os.path.splitext(os.path.basename(args.train_file))[0]
    data_dir = os.path.dirname(args.train_file)
    config_info = f"_seed_{args.split_seed}-lr_{args.lr}-epoch_{args.epoch}-batch_{args.batch_size}_{args.a1}_{args.a2}_{args.a3}_{args.a4}_{args.a5}_{args.a6}-dacc_{args.backtrack_dacc}-dF1_{args.backtrack_dF1}"

    assert args.opt_metrics == 'acc' or args.opt_metrics == 'F1'

    model_path = myconfig.model_base_path + f'model-opt_{args.opt_metrics}/model-{train_filename}/depth_{args.net_depth}-width_{args.net_width}-iterations_{args.iterations}-timeout_{args.timeout}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file = model_path + f'model_{config_info}'


    log_path = myconfig.log_base_path + f'log-opt_{args.opt_metrics}/log-{train_filename}/depth_{args.net_depth}-width_{args.net_width}-iterations_{args.iterations}-timeout_{args.timeout}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = log_path + f'log_{config_info}'


    result_path = myconfig.result_base_path + f'result-opt_{args.opt_metrics}/result-{train_filename}/{args.mode}/depth_{args.net_depth}-width_{args.net_width}-iterations_{args.iterations}-timeout_{args.timeout}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_file = result_path + f'result_{config_info}'

    split_data_path = None
    if args.mode == 'train-test':
        if args.ratio!=1.0 and args.rest==True:
            # split_data_path = f'{data_dir}/{args.ratio}-train-test_{train_filename}/'      # train_test_data_path
            split_data_path = f'{data_dir}/train-test_{train_filename}/'      # train_test_data_path
            if not os.path.exists(split_data_path):
                os.makedirs(split_data_path)
    elif args.mode == 'k-fold':
        split_data_path = f'{data_dir}/{args.kfold}-fold_{train_filename}/'                 # kfold_data_path
        if not os.path.exists(split_data_path):
            os.makedirs(split_data_path)

    return model_file, log_file, result_file, split_data_path




def store_result(result_file, info_names, info_values, args, appendix=None):

    assert len(info_names)==len(info_values)
    info_names.append("\n***** The args of experiment")
    info_values.append(" *****")

    for arg in vars(args):
        info_names.append(arg)
        info_values.append(getattr(args, arg))

    with open(result_file, 'w') as f:
        for name, value in zip(info_names, info_values):
            f.write(f"{name} : {value}\n")
        if appendix is not None:
            f.write("\n*****Appendix:*****\n")
            f.write(str(appendix))


def get_train_args_list(args, train_file):


    train_args_list = ['--train_file', train_file, '--model_file', args.model_file, '--log_file', args.log_file, '--device', args.device, '--timeout',  str(args.timeout), '--opt_metrics',  args.opt_metrics,   \
        '--tau_start', str(args.tau_start),   '--tau_end', str(args.tau_end),  '--backtrack_dF1', str(args.backtrack_dF1), '--backtrack_dacc', str(args.backtrack_dacc), '--random_seed', str(args.random_seed) ,  \
        '--lr', str(args.lr) , '--iterations', str(args.iterations),  '--a1', str(args.a1),      '--a2', str(args.a2),  '--a3', str(args.a3),    '--a4', str(args.a4),  '--a5', str(args.a5),    '--a6', str(args.a6), \
        '--batch_size', str(args.batch_size) ,    '--epoch', str(args.epoch),    '--net_depth', str(args.net_depth),       '--net_width', str(args.net_width), '--threshold', str(args.threshold)]

    if args.atoms_chosen is not None:
        train_args_list.append('--atoms_chosen')
        train_args_list.append(args.atoms_chosen)
    if args.tau_decay is not None:
        train_args_list.append('--tau_decay')
        train_args_list.append(str(args.tau_decay))

    return train_args_list



def get_k_fold_split(dataset, k_fold, save_indices_file, split_seed):

    fold_indices = []
    if os.path.exists(save_indices_file):
        print(f'load {k_fold}-fold data from {save_indices_file} ...')
        with open(save_indices_file, 'r') as f:
            test_indices_list = f.readlines()
            test_indices_list = [line.strip().split(',') for line in test_indices_list]
            test_indices_list = [[int(num) for num in line] for line in test_indices_list]
            for k in range(k_fold):
                # temp_list = []
                kth_train_indices = []
                for j in range(k_fold):
                    if j!=k:
                        kth_train_indices += test_indices_list[j]
                # print(kth_train_indices)
                kth_fold = (kth_train_indices, test_indices_list[k])
                fold_indices.append(kth_fold)
            # print(fold_indices)

    else:

        kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=split_seed)
        data = dataset.iloc[:, 0:-1]
        label =  dataset.iloc[:, -1]
        fold_indices = list(kf.split(X=data, y=label))    # Stratification is done based on the y labels.
        _fold_indices = fold_indices
        print(f'generate {k_fold}-fold data using StratifiedKFold... {save_indices_file}')
        with open(save_indices_file, 'w') as fo:
            for i, (_, test_index ) in enumerate(_fold_indices): 
                fo.write(", ".join(list(map(str, test_index))))
                fo.write("\n")
        # exit()

    return fold_indices





def run_k_fold_cross_validation(args, k_fold):

    model_file, log_file, result_file, kfold_data_path = prepare_path(args)
    args.model_file = model_file
    args.log_file = log_file
    args.test_file = None
    args.ratio = None
    args.rest = None

    data_filename = os.path.splitext(os.path.basename(args.train_file))[0]
    split_data_info = f"{k_fold}fold_indices-{data_filename}_{args.split_seed}"
    save_indices_file = kfold_data_path + f"{split_data_info}.indices"

    dataset = pd.read_csv(args.train_file)
    fold_indices = get_k_fold_split(dataset, k_fold, save_indices_file, args.split_seed)

    train_time_list, is_ordered_list, nodes_num_list, BDDs_s, ops_s = [], [], [], [], []
    bdd_best_s, bdd_root_best_s = [], []
    net_train_acc_list, net_train_F1_macro_list, net_train_F1_list, net_train_prec_list, net_train_rec_list = [], [], [], [], []
    net_test_acc_list, net_test_F1_macro_list, net_test_F1_list, net_test_prec_list, net_test_rec_list = [], [], [], [], []

    train_acc_list, train_F1_macro_list, train_F1_list, train_prec_list, train_rec_list = [], [], [], [], []
    test_acc_list, test_F1_macro_list, test_F1_list, test_prec_list, test_rec_list = [], [], [], [], []

    print(f"\n************ {k_fold}_fold_cross_validation **********")

    for i, (train_index, test_index) in enumerate(fold_indices):
        print(f'\n-------------- {i}th-fold ---------------')
        train_data = dataset.iloc[train_index]
        test_data = dataset.iloc[test_index]

        save_train_file = kfold_data_path + f"{split_data_info}_train_fold_{i}.csv"     # store the i-th fold dataset
        save_test_file = kfold_data_path + f"{split_data_info}_test_fold_{i}.csv"
        if not os.path.exists(save_train_file):
            print(f'generate train data: {save_train_file} ...')
            train_data.to_csv(save_train_file, index=False)
        if not os.path.exists(save_test_file):
            print(f'generate test data: {save_test_file} ...')
            test_data.to_csv(save_test_file, index=False)


        train_args_list = get_train_args_list(args, train_file=save_train_file)
        train_args = train.get_train_argument(train_args_list)
        train_time, net_train_acc, net_train_F1_macro, net_train_F1, net_train_prec, net_train_rec, \
        train_acc, train_F1_macro, train_F1, train_prec, train_rec, is_ordered, nodes_num, \
        BDD_s, apply_s, bdd_s, bdd_root_s, bdd_best, bdd_root_best = train.train(train_args)

        test_args = test.get_test_argument(['--test_file', save_test_file, 
        '--model_file', model_file, 
        '--threshold', str(args.threshold), 
        '--device', args.device])
        net_test_acc, net_test_F1_macro, net_test_F1, net_test_prec, net_test_rec, \
        _, _, _, _, _, _ , _ = test.test(test_args)    # NOTE: the result we get here is just the performance of the neural network model trained in the 0-th iteration

        # NOTE: nodes_num returned by test.test() is the size of trained bdd model at the 0-th iteration; 
        #       instead, nodes_num returned by test.predict_score_with_bdd() is the size of the final best bdd model
        test_acc, test_F1_macro, test_F1, test_prec, test_rec, nodes_num = test.predict_score_with_bdd(save_test_file, bdd_best, bdd_root_best, args.device)



        os.remove(model_file)
        # os.remove(log_file)
        # os.remove(result_file)
        # if os.path.exists(save_train_file):      
        #     os.remove(save_train_file)
        # if os.path.exists(save_test_file):
        #     os.remove(save_test_file)

        train_time_list.append(train_time)
        is_ordered_list.append(is_ordered)
        nodes_num_list.append(nodes_num)  
        # BDD_s.append(BDD)      
        BDDs_s.append(BDD_s)    
        ops_s.append(apply_s)
        bdd_best_s.append(bdd_best)
        bdd_root_best_s.append(bdd_root_best)

        net_train_acc_list.append(net_train_acc)
        net_train_F1_macro_list.append(net_train_F1_macro)
        net_train_F1_list.append(net_train_F1)
        net_train_prec_list.append(net_train_prec)
        net_train_rec_list.append(net_train_rec)
        net_test_acc_list.append(net_test_acc)
        net_test_F1_macro_list.append(net_test_F1_macro)
        net_test_F1_list.append(net_test_F1)
        net_test_prec_list.append(net_test_prec)
        net_test_rec_list.append(net_test_rec)

        train_acc_list.append(train_acc)
        train_F1_macro_list.append(train_F1_macro)
        train_F1_list.append(train_F1)
        train_prec_list.append(train_prec)
        train_rec_list.append(train_rec)
        test_acc_list.append(test_acc)
        test_F1_macro_list.append(test_F1_macro)
        test_F1_list.append(test_F1)
        test_prec_list.append(test_prec)
        test_rec_list.append(test_rec)


    avg_train_time = sum(train_time_list) / len(train_time_list)
    are_all_ordered =  all(is_ordered_list)
    
    # avg_size of the final reduced OBDDs that are produced by the dd package and then coverted into the simple form of OBDD defined in the paper
    # (NOTE: hence they do not contain complement edges any more)
    avg_nodes_num = sum(nodes_num_list) / len(nodes_num_list)       

    avg_net_train_acc = sum(net_train_acc_list) / len(net_train_acc_list)
    avg_net_train_F1_macro = avg_list(net_train_F1_macro_list) 
    avg_net_train_F1 = sum(net_train_F1_list) / len(net_train_F1_list)
    avg_net_train_prec = sum(net_train_prec_list) / len(net_train_prec_list)
    avg_net_train_rec = sum(net_train_rec_list) / len(net_train_rec_list)
    avg_net_test_acc = sum(net_test_acc_list) / len(net_test_acc_list)
    avg_net_test_F1_macro = avg_list(net_test_F1_macro_list) 
    avg_net_test_F1 = sum(net_test_F1_list) / len(net_test_F1_list)
    avg_net_test_prec = sum(net_test_prec_list) / len(net_test_prec_list)
    avg_net_test_rec = sum(net_test_rec_list) / len(net_test_rec_list)

    avg_train_acc = sum(train_acc_list) / len(train_acc_list)
    avg_train_F1_macro = avg_list(train_F1_macro_list) 
    avg_train_F1 = sum(train_F1_list) / len(train_F1_list)
    avg_train_prec = sum(train_prec_list) / len(train_prec_list)
    avg_train_rec = sum(train_rec_list) / len(train_rec_list)
    avg_test_acc = sum(test_acc_list) / len(test_acc_list)
    avg_test_F1_macro = avg_list(test_F1_macro_list) 
    avg_test_F1 = sum(test_F1_list) / len(test_F1_list)
    avg_test_prec = sum(test_prec_list) / len(test_prec_list)
    avg_test_rec = sum(test_rec_list) / len(test_rec_list)


    avg_size_nonreducedOBDD = -1    # avg_size of the original OBDDs produced by OBDD-NET (non-reduced)
    size_nonreducedOBDD_list = []
    if args.iterations == 1:
        cnt_temp = 0
        # for fold_i, BDDs in enumerate(BDDs_s):
        #     # assert len(BDDs)==1
        #     cnt_temp += BDDs[0].get_vertices_cnt()
        # avg_size_nonreducedOBDD = cnt_temp/len(BDDs_s)
        for fold_i, BDDs in enumerate(BDDs_s):
            # assert len(BDDs)==1
            _size = BDDs[0].get_vertices_cnt()
            size_nonreducedOBDD_list.append( _size )
        avg_size_nonreducedOBDD = sum(size_nonreducedOBDD_list)/len(BDDs_s)


    print(f'\navg_train_time: {avg_train_time} | avg_net_train_acc: {avg_net_train_acc} | avg_net_train_F1_macro: {avg_net_train_F1_macro} | avg_net_train_F1: {avg_net_train_F1} \
        \navg_train_acc: {avg_train_acc} | avg_train_F1_macro: {avg_train_F1_macro} | avg_train_F1: {avg_train_F1} | avg_test_acc: {avg_test_acc} | avg_test_F1_macro: {avg_test_F1_macro} | avg_test_F1: {avg_test_F1} |\
        \nare_all_ordered: {are_all_ordered} | avg_size_nonreducedOBDD: {avg_size_nonreducedOBDD} | avg_size_reducedOBDD: {avg_nodes_num}')

    # info_names = ['avg_train_time', 'avg_net_train_acc', 'avg_net_train_F1', 'avg_train_acc', 'avg_train_F1_score', 'are_all_ordered', 'avg_BDD_size', 'avg_test_acc', 'avg_test_F1_score']
    info_names = ['avg_train_time', 'avg_net_train_acc', 'avg_net_train_F1_macro', 'avg_net_train_F1', 'avg_net_train_prec', 'avg_net_train_rec',\
        'avg_net_test_acc', 'avg_net_test_F1_macro', 'avg_net_test_F1', 'avg_net_test_prec', 'avg_net_test_rec',\
        'avg_train_acc', 'avg_train_F1_macro', 'avg_train_F1', 'avg_train_prec', 'avg_train_rec',\
        'avg_test_acc', 'avg_test_F1_macro', 'avg_test_F1', 'avg_test_prec', 'avg_test_rec',\
        'are_all_ordered', 'avg_size_nonreducedOBDD', 'avg_size_reducedOBDD']
    info_values = [avg_train_time, avg_net_train_acc, avg_net_train_F1_macro, avg_net_train_F1, avg_net_train_prec, avg_net_train_rec,\
        avg_net_test_acc, avg_net_test_F1_macro, avg_net_test_F1, avg_net_test_prec, avg_net_test_rec,\
        avg_train_acc, avg_train_F1_macro, avg_train_F1, avg_train_prec, avg_train_rec,\
        avg_test_acc, avg_test_F1_macro, avg_test_F1, avg_test_prec, avg_test_rec,\
        str(are_all_ordered), avg_size_nonreducedOBDD, avg_nodes_num]

    detailkfold_info_names = ['kfold_train_time', 'kfold_net_train_acc', 'kfold_net_train_F1_macro', 'kfold_net_train_F1', 'kfold_net_train_prec', 'kfold_net_train_rec',\
        'kfold_net_test_acc', 'kfold_net_test_F1_macro', 'kfold_net_test_F1', 'kfold_net_test_prec', 'kfold_net_test_rec',\
        'kfold_train_acc', 'kfold_train_F1_macro', 'kfold_train_F1', 'kfold_train_prec', 'kfold_train_rec',\
        'kfold_test_acc', 'kfold_test_F1_macro', 'kfold_test_F1', 'kfold_test_prec', 'kfold_test_rec',\
        'kfold_is_ordered', 'kfold_size_nonreducedOBDD', 'kfold_size_reducedOBDD']
    detailkfold_info_values = [train_time_list, net_train_acc_list, net_train_F1_macro_list, net_train_F1_list, net_train_prec_list, net_train_rec_list,\
        net_test_acc_list, net_test_F1_macro_list, net_test_F1_list, net_test_prec_list, net_test_rec_list,\
        train_acc_list, train_F1_macro_list, train_F1_list, train_prec_list, train_rec_list,\
        test_acc_list, test_F1_macro_list, test_F1_list, test_prec_list, test_rec_list,\
        str(is_ordered_list), size_nonreducedOBDD_list, nodes_num_list]

    info_names += detailkfold_info_names
    info_values += detailkfold_info_values

    rule_info = "\n-----The JSON descriptions of best bdd rule:-----\n"
    for fold_i, bdd_best in enumerate(bdd_best_s):
        rule_info += f"\n---In the {fold_i}-th fold:---\n"
        rule_info += util.bdd_dump_str(bdd_best, bdd_root_best_s[fold_i])

    rule_info += "\n\n-----The (dot) descriptions of interpreted BDD:-----\n"
    for fold_i, BDDs in enumerate(BDDs_s):
        rule_info += f"\n---In the {fold_i}-th fold:---\n"
        rule_info += f"\nthe sequence of operator: " + ', '.join(ops_s[fold_i])
        for iter_j, BDD in enumerate(BDDs):
            rule_info += f"\n\n--the {iter_j}-th iteration:--\n" + BDD.get_dot_description()

    store_result(result_file, info_names, info_values, args, appendix=rule_info)

    
def get_train_test_split(dataset, ratio, save_indices_file, split_seed):
    if os.path.exists(save_indices_file):
        print(f'load {ratio}-train_test data from {save_indices_file} ...')
        with open(save_indices_file, 'r') as f:
            train_test_indices = f.readlines()
            train_test_indices = [line.strip().split(',') for line in train_test_indices]
            train_test_indices = [[int(num) for num in line] for line in train_test_indices]
            train_index = train_test_indices[0]
            test_index = train_test_indices[1]
            print('len(train_index)+len(test_index) :', len(train_index)+len(test_index))
    else:
        # Initialize the cross-validation object
        data = dataset.iloc[:, 0:-1]
        label =  dataset.iloc[:, -1]
        # train_data, train_label, test_data, test_label = train_test_split(data, label, train_size=ratio, random_state=split_seed, stratify=label) # Stratification is done based on the label.
        sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=split_seed) # get one train_test split
        train_test_indices = sss.split(X=data, y=label)    # Stratification is done based on the y labels.
        # _train_test_indices = train_test_indices
        print(f'generate {ratio}-train_test data using StratifiedShuffleSplit... {save_indices_file}')
        # exit()
        with open(save_indices_file, 'w') as fo:
            for train_index, test_index in train_test_indices: 
                fo.write(", ".join(list(map(str, train_index))))
                fo.write("\n")
                fo.write(", ".join(list(map(str, test_index))))
                fo.write("\n")

    # return train_test_indices
    return train_index, test_index



def run_train_test(args):       

    model_file, log_file, result_file, train_test_data_path = prepare_path(args)
    args.model_file = model_file
    args.log_file = log_file
    args.kfold = None

    print("\n***** train & test *****")
    train_file, test_file = None, None
    if args.ratio == 1.0 or args.rest == False:
        assert args.test_file is not None
        args.rest = False
        
        train_file = args.train_file
        test_file = args.test_file
    else:
        data_filename = os.path.splitext(os.path.basename(args.train_file))[0]
        # data_dir = os.path.dirname(args.train_file)
        split_data_info = f"train_test-{data_filename}_{args.ratio}_{args.split_seed}"

        # the file to store the split indices, training data and testing data
        save_indices_file = train_test_data_path + f"{split_data_info}.indices"
        save_train_file = train_test_data_path + f"{split_data_info}_train.csv"     
        save_test_file = train_test_data_path + f"{split_data_info}_test.csv"

        dataset = pd.read_csv(args.train_file)
        # apply StratifiedShuffleSplit to split the dataset
        train_index, test_index = get_train_test_split(dataset, args.ratio, save_indices_file, args.split_seed)

        # store the training data and testing data
        train_data = dataset.iloc[train_index]
        test_data = dataset.iloc[test_index]

        if not os.path.exists(save_train_file):
            print(f'generate train data: {save_train_file} ...')
            train_data.to_csv(save_train_file, index=False)
        if not os.path.exists(save_test_file):
            print(f'generate test data: {save_test_file} ...')
            test_data.to_csv(save_test_file, index=False)

        train_file = save_train_file
        test_file = save_test_file

    train_args_list = get_train_args_list(args, train_file=train_file)
    train_args = train.get_train_argument(train_args_list)
    train_time, net_train_acc, net_train_F1_macro, net_train_F1, net_train_prec, net_train_rec, \
    train_acc, train_F1_macro, train_F1, train_prec, train_rec, is_ordered, nodes_num, \
    BDD_s, apply_s, bdd_s, bdd_root_s, bdd_best, bdd_root_best = train.train(train_args)

    test_args = test.get_test_argument(['--test_file', test_file, 
    '--model_file', model_file, 
    '--threshold', str(args.threshold), 
    '--device', args.device])
    net_test_acc, net_test_F1_macro, net_test_F1, net_test_prec, net_test_rec, \
    _, _, _, _, _, _ , _ = test.test(test_args)    # NOTE: the result we get here is just the performance of the neural network model trained in the 0-th iteration

    # NOTE: nodes_num returned by test.test() is the size of trained bdd model at the 0-th iteration; 
    #       instead, nodes_num returned by test.predict_score_with_bdd() is the size of the final best bdd model
    # nodes_num we get here is the size of the final reduced OBDD that is produced by the dd package and then coverted into the simple form of OBDD defined in the paper
    # (NOTE: hence it does not contain complement edges any more)
    test_acc, test_F1_macro, test_F1, test_prec, test_rec, nodes_num = test.predict_score_with_bdd(test_file, bdd_best, bdd_root_best, args.device)

    os.remove(model_file)
    # os.remove(log_file)
    # os.remove(result_file)
    # if args.ratio!=1.0 and args.rest==True:
        # if os.path.exists(train_file):
        #     os.remove(train_file)
        # if os.path.exists(test_file):
        #     os.remove(test_file)

    size_nonreducedOBDD = -1    # size of the original OBDDs produced by OBDD-NET (non-reduced)
    if args.iterations == 1:
        # assert len(BDDs)==1
        size_nonreducedOBDD = BDD_s[0].get_vertices_cnt()

    print(f'\ntrain_time: {train_time} | net_train_acc: {net_train_acc} | net_train_F1_macro: {net_train_F1_macro} | net_train_F1: {net_train_F1} \
        \ntrain_acc: {train_acc} | train_F1_macro: {train_F1_macro} | train_F1: {train_F1} | test_acc: {test_acc} | test_F1_macro: {test_F1_macro} | test_F1: {test_F1} |\
        \nis_ordered: {is_ordered} | size_nonreducedOBDD: {size_nonreducedOBDD}  | size_reducedOBDD: {nodes_num}')

    info_names = ['train_time', 'net_train_acc', 'net_train_F1_macro', 'net_train_F1', 'net_train_prec', 'net_train_rec',\
        'net_test_acc', 'net_test_F1_macro', 'net_test_F1', 'net_test_prec', 'net_test_rec',\
        'train_acc', 'train_F1_macro', 'train_F1', 'train_prec', 'train_rec',\
        'test_acc', 'test_F1_macro', 'test_F1', 'test_prec', 'test_rec',\
        'is_ordered', 'size_nonreducedOBDD', 'size_reducedOBDD']
    info_values = [train_time, net_train_acc, net_train_F1_macro, net_train_F1, net_train_prec, net_train_rec,\
        net_test_acc, net_test_F1_macro, net_test_F1, net_test_prec, net_test_rec,\
        train_acc, train_F1_macro, train_F1, train_prec, train_rec,\
        test_acc, test_F1_macro, test_F1, test_prec, test_rec,\
        str(is_ordered), size_nonreducedOBDD, nodes_num]


    rule_info = "\n-----The JSON descriptions of best bdd rule:-----\n"
    rule_info += util.bdd_dump_str(bdd_best, bdd_root_best)

    rule_info += "\n\n-----The (dot) descriptions of interpreted BDD:-----\n"
    rule_info += f"\nthe sequence of operator: " + ', '.join(apply_s)
    for iter_j, BDD in enumerate(BDD_s):
        rule_info += f"\n\n--the {iter_j}-th iteration:--\n" + BDD.get_dot_description()


    store_result(result_file, info_names, info_values, args, appendix=rule_info)




def get_experiment_argument(args_list=None):
    experiment_parser = argparse.ArgumentParser(description='Main script for experiment')
    experiment_parser.add_argument("--mode", choices=["k-fold", "train-test"], required=False, default="k-fold", help="experiment mode")
    experiment_parser.add_argument('--kfold', type=int, required=False, default=5, help='the number of folds, when the mode is set \"k-fold\"')
    experiment_parser.add_argument('--ratio', type=float, required=False, default=1.0, help='the ratio of the dataset to include in the train split, when the mode is set \"train-test\"')
    experiment_parser.add_argument('--rest', type=bool, required=False, default=False, help='wheather to use the rest of split dataset as testing data, when the mode is set \"train-test\"')
    experiment_parser.add_argument('--train_file', type=str, required=True)
    experiment_parser.add_argument('--test_file', type=str, required=False, default=None) 
    experiment_parser.add_argument('--model_file', type=str, required=False, default='model/train_model')
    experiment_parser.add_argument('--log_file', type=str, required=False, default='log/train_log')

    experiment_parser.add_argument('--net_depth', type=int, required=False, default=6, help='(maximal) number of decision levels of the learned OBDD')
    experiment_parser.add_argument('--net_width', type=int, required=False, default=None, help='(maximal) number of nodes in each decision level. Default value:  2^(net_depth-1)')
    experiment_parser.add_argument('--epoch', type=int, required=False, default=3000, help='epoch number')
    experiment_parser.add_argument('--lr', type=float, required=False, default=0.2, help='learn rate')
    experiment_parser.add_argument('--iterations', type=int, required=False, default=1, help='the number of iterations (during iterative learning)')
    experiment_parser.add_argument('--a1', type=float, required=False, default=1, help='regular cof. towards read-once poperty')
    experiment_parser.add_argument('--a2', type=float, required=False, default=1e-6, help='regular cof. towards 0-1-binarization for params enc_dec')
    experiment_parser.add_argument('--a3', type=float, required=False, default=1e-6, help='regular cof. towards 0-1-binarization for params enc_lef & enc_right')
    experiment_parser.add_argument('--a4', type=float, required=False, default=0, help='regular cof. towards size optimization')
    experiment_parser.add_argument('--a5', type=float, required=False, default=1.1, help='regular cof. towards the loss weight of the opposite class of the prediction to be refined in iterative learning')
    experiment_parser.add_argument('--a6', type=float, required=False, default=0, help='regular cof. towards persisting the compatibility with the given ordering: ordering_to_compatible during iterative learning')
    experiment_parser.add_argument('--batch_size', type=int, required=False, default=512, help='batch size')
    experiment_parser.add_argument('--random_seed', type=int, required=False, default=2024, help='random seed (for model initalization)')
    experiment_parser.add_argument('--split_seed', type=int, required=False, default=2024, help='random seed (for splitting dataset)')
    experiment_parser.add_argument('--threshold', type=float, required=False, default=0.5, help='threshold for classification')
    experiment_parser.add_argument('--atoms_chosen', type=str, required=False, default=None, help='specify the indices of atoms chosen. e.g, \'0, 2, 5, 3\'')
    
    experiment_parser.add_argument("--opt_metrics", choices=["acc", "F1"], default="acc", help="specify the optimization metrics used to select mode during training: acc or (macro) F1")
    experiment_parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device')
    experiment_parser.add_argument('--timeout', type=int, default=900, help='timeout for training process (unit: s)')
    experiment_parser.add_argument('--tau_start', type=float, default=10, help='the initial value of the temperature parameter tau in Gumbel Softmax function')
    experiment_parser.add_argument('--tau_end', type=float, default=1, help='the terminal value of the temperature parameter tau')
    experiment_parser.add_argument('--tau_decay', type=float, default=None, help='the decay factor of thetemperature parameter tau. if None, an proper value will be automatically taken based on epoch, timeout and iterations')
    experiment_parser.add_argument('--backtrack_dF1', type=float, default=0.3, help='the (macro) F1 performance degradation threshold to trigger backtracking to the acquired best model')
    experiment_parser.add_argument('--backtrack_dacc', type=float, default=0.3, help='the acc performance degradation threshold to trigger backtracking to the acquired best model')

    if args_list==None:
        experiment_args = experiment_parser.parse_args()
    else:
        experiment_args = experiment_parser.parse_args(args_list)
    return experiment_args



if __name__ == "__main__":

    args = get_experiment_argument()

   
    if args.mode == "k-fold":
        # assert args.kfold is not None
        # args.test_file = None
        # args.ratio = None
        # args.rest = None
        if args.net_width is None:
            args.net_width = 2**(args.net_depth - 1)
        run_k_fold_cross_validation(args, k_fold=args.kfold)


    elif args.mode == "train-test":
        # args.kfold = None
        # if args.ratio == 1.0 or args.rest == False:
        #     assert args.test_file is not None
        #     args.rest = False
        if args.net_width is None:
            args.net_width = 2**(args.net_depth - 1)
        run_train_test(args)
        pass
