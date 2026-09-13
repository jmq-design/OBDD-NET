import torch
import math
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import os
import json
import pandas as pd
import argparse
import time
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

from model import OBDDNet
from data import AssignmentDataset
from utils.bdd import BinaryDecisionDiagram

import myconfig
import utils.util as util


def test(test_args):
    device=test_args.device
    test_file = test_args.test_file
    model_file = test_args.model_file
    threshold = test_args.threshold

    model_params = torch.load(model_file)

    print('\n>testing with the trained neural network model...\t ', test_file)

    test_dataset = AssignmentDataset(test_file, device)
    dataset_size, atom_size, pos_distribution, minority_class = util.dataset_statistic(test_dataset)
    print(f"Dataset Statistic: dataset_size: {dataset_size} | atom_size: {atom_size} | pos_distribution:{pos_distribution} | minority_class: {minority_class}")


    net_depth = len(model_params['params_dec'])
    net_width = len(model_params['params_left.1'])
    print('net_depth: %d | net_width: %d'%(net_depth, net_width))


    model = OBDDNet(net_width, net_depth, atom_size)
    model.to(device)
    model.load_state_dict(model_params)
    model.eval()


    torch.set_printoptions(threshold=1000, precision=3, sci_mode=False) 

    # net_test_acc, net_test_F1, net_test_prec, net_test_rec = model.score(test_dataset, model_type='train-original') 
    # print('For train-original model: \t net_test_acc: %.3f | (net_test_F1, net_test_prec, net_test_rec):( %.3f/ %.3f/ %.3f)'%(net_test_acc, net_test_F1, net_test_prec, net_test_rec))
    net_test_acc, net_test_F1_macro, net_test_F1, net_test_prec, net_test_rec = model.score(test_dataset, model_type='test-original', average=None) 
    # print('For test-original model: \t net_test_acc: %.3f | (net_test_F1, net_test_prec, net_test_rec):( %.3f/ %.3f/ %.3f)'%(net_test_acc, net_test_F1, net_test_prec, net_test_rec))
    # test_acc, test_F1, test_prec, test_rec, is_ordered, nodes_num = model.score(test_dataset, model_type='interpreted') 
    # print('test_acc: %.3f | (test_F1, test_prec, test_rec):( %.3f/ %.3f/ %.3f) | is_ordered: %s | nodes_num: %d'%(test_acc, test_F1, test_prec, test_rec, str(is_ordered), nodes_num))

    test_acc, test_F1_macro, test_F1, test_prec, test_rec, is_ordered, nodes_num = model.score(test_dataset, model_type='interpreted',average=None) 
    print(f"For trained model at the 0th iteration: \n net_test_acc: {net_test_acc} | net_test_F1_macro: {net_test_F1_macro} | net_test_F1: {net_test_F1} | net_test_prec:{net_test_prec} | net_test_rec: {net_test_rec}\n" \
    + f" | test_acc: {test_acc} | test_F1_macro: {test_F1_macro} | test_F1: {test_F1} | test_prec:{test_prec} | test_rec: {test_rec} | is_ordered: {is_ordered} | nodes_num (for BDD): {nodes_num}")
    
    return net_test_acc, net_test_F1_macro, net_test_F1, net_test_prec, net_test_rec, test_acc, test_F1_macro, test_F1, test_prec, test_rec, is_ordered, nodes_num


def predict_score_with_bdd(save_test_file, bdd, bdd_root, device):
    print(f"\n>testing with the interpreted bdd model...\t {save_test_file}")
    test_dataset = AssignmentDataset(save_test_file, device)
    # dataset_size, feature_cnt, pos_distribution, minority_class = util.dataset_statistic(test_dataset)
    # print(f"Dataset Statistic: dataset_size: {dataset_size} | feature_cnt: {feature_cnt} | pos_distribution:{pos_distribution} | minority_class: {minority_class}")
    
    pred, test_acc, test_F1_macro, test_F1, test_prec, test_rec = util.bdd_predict_score(test_dataset, bdd, bdd_root)
    # nodes_num = len(bdd)  # as it contains complement edge, we do not use it!!
    nodes_num = util.bdd_actual_size(bdd, bdd_root) 

    print(f"For final best bdd model: \n test_acc: {test_acc} | test_F1_macro: {test_F1_macro} | test_F1: {test_F1} | test_prec:{test_prec} | test_rec: {test_rec} | nodes_num: {nodes_num}")

    return test_acc, test_F1_macro, test_F1, test_prec, test_rec, nodes_num




def get_F1(prediction, label):

    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()


    F1_score = f1_score(label, prediction)
    precision = precision_score(label, prediction)
    recall = recall_score(label, prediction)

    return F1_score, precision, recall




def get_test_argument(args_list=None):
    test_parser = argparse.ArgumentParser(description='Main script for test')
    test_parser.add_argument('--test_file', type=str, required=True)
    test_parser.add_argument('--model_file', type=str, required=False, default='model/train_model')
    test_parser.add_argument('--threshold', type=float, required=False, default=0.5, help='threshold for classification')
    test_parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device')

    if args_list==None:
        test_args = test_parser.parse_args()
    else:
        test_args = test_parser.parse_args(args_list)
    return test_args






