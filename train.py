import torch
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import time
import sys
import copy
# import graphviz

import numpy as np
from model import OBDDNet
from data import AssignmentDataset
# import myconfig
from test import get_F1
import utils.util as util

# device = myconfig.DEVICE


LOG = False
# LOG = True
TIMESTAMP = "{0:_%Y-%m-%d_%H-%M-%S/}".format(datetime.now())

avg_list = lambda val_s: sum(val_s) / len(val_s)



def refine_guide(original_train_dataset, bdd, bdd_root):
    

    train_dataset = copy.deepcopy(original_train_dataset)


    stime_bdd_pred = time.time()
    pred, train_acc, train_F1_macro, train_F1, train_prec, train_rec = util.bdd_predict_score(train_dataset, bdd, bdd_root)
    print(f"bdd_predict_score uses time : {time.time()-stime_bdd_pred}s")    
    
    print(f"\nFor the current (refined) bdd: \n train_acc: {train_acc} | train_F1_macro: {train_F1_macro} | train_F1: {train_F1} | train_prec:{train_prec} | train_rec: {train_rec}")

    prediction_to_refine, op_for_refine = None, None
    

    train_prec_rec = list(train_prec) + list(train_rec)
    min_score = min(train_prec_rec)
    min_id = train_prec_rec.index(min_score)
    if (min_id == 1) or (min_id == 2):      # prec. for the class 1 / rec. for the class 0
        prediction_to_refine = 1 
        op_for_refine = "AND"
        selected_indices = [index for index, value in enumerate(pred) if value == 1]
    elif (min_id == 0) or (min_id == 3):    # prec. for the class 0 / rec. for the class 1
        prediction_to_refine = 0
        op_for_refine = "OR"
        selected_indices = [index for index, value in enumerate(pred) if value == 0]
    else:
        raise ValueError()    




    train_dataset.data = train_dataset.data[selected_indices]
    train_dataset.labels = train_dataset.labels[selected_indices]

    features = util.get_bdd_feature_ordering(bdd)
    print(f" The features declared in current bdd: {features}")

    feature_ordering = util.get_bdd_feature_ordering__support(bdd, bdd_root)
    print(f" The feature ordering of current bdd: {feature_ordering}")


    ordering_to_compatible = str(feature_ordering).strip("[]")

    return train_dataset, prediction_to_refine, op_for_refine, ordering_to_compatible



def dump_BDD_s(BDD_s, save_file="BDD_s.dot"):
    with open(save_file, "w") as fo:
        for BDD in BDD_s:
            dot_des = BDD.get_dot_description() 
            fo.write(dot_des + "\n")



def print_bdds_info(dataset, bdd, bdd_roots, opt_metrics):

    dataset_size, feature_cnt, pos_distribution, minority_class = util.dataset_statistic(dataset)
    print(f"\ndataset_statistic: \n dataset_size: {dataset_size} | feature_cnt: {feature_cnt} | pos_distribution:{pos_distribution} | minority_class: {minority_class}")

    # bdd.dump('bdd_roots.png', roots=bdd_roots)

    bdd_s, bdd_root_s = [], []
    bdd_best, bdd_root_best = None, None
    best_acc, best_F1 = -1, -1
    best_model_acc, best_model_F1 = -1, -1

    print(f"\n** Comparison of the resulting bdd models at each iteration:")

    for idx, bdd_root in enumerate(bdd_roots):

        bdd_, bdd_root_ = util.copy_bdd(bdd, bdd_roots, bdd_roots[idx])
        nodes_num = util.bdd_actual_size(bdd_, bdd_root_)
        feature_ordering = util.get_bdd_feature_ordering__support(bdd_, bdd_root_)
        pred, train_acc, train_F1_macro, train_F1, train_prec, train_rec = util.bdd_predict_score(dataset, bdd_, bdd_root_)
        
        # train_F1_macro = avg_list(train_F1)

        print(f"\n [TOCHECK] The resulting bdd from the first {idx} iterations: ")
        # print(f"prediction: {pred}")
        print(f"train_acc: {train_acc} | train_F1_macro: {train_F1_macro} | train_F1: {train_F1} | train_prec:{train_prec} | train_rec: {train_rec}")
        # print(f"feature_ordering: {feature_ordering} | nodes_num (negated edge is applied!!): {nodes_num}")
        print(f"feature_ordering: {feature_ordering} | nodes_num: {nodes_num}")

        bdd_s.append(bdd_)
        bdd_root_s.append(bdd_root_)


        if opt_metrics == 'F1':
            if (train_F1_macro > best_F1) or ((train_F1_macro == best_F1) and (train_acc>best_model_acc)): 
                best_F1, best_model_acc = train_F1_macro, train_acc
                bdd_best, bdd_root_best = bdd_, bdd_root_
            else:
                print(f"!! Not the best so far: the {idx}-th iteration.")


        elif opt_metrics == 'acc':
            if (train_acc>best_acc) or ((train_acc == best_acc) and (train_F1_macro > best_model_F1)): 
                best_acc, best_model_F1 = train_acc, train_F1_macro
                bdd_best, bdd_root_best = bdd_, bdd_root_
            else:
                print(f"!! Not the best so far: the {idx}-th iteration.")

    return bdd_s, bdd_root_s, bdd_best, bdd_root_best





def train(train_args):
    device=train_args.device
    min_loss = 5e-5

    # load the train_args
    train_file = train_args.train_file
    log_file = train_args.log_file
    model_dir = train_args.model_file
    net_width = train_args.net_width
    net_depth = train_args.net_depth
    batch_size = train_args.batch_size
    epoch_num = train_args.epoch
    iterations = train_args.iterations
    learn_rate = train_args.lr
    threshold = train_args.threshold
    atoms_chosen = train_args.atoms_chosen
    timeout = train_args.timeout
    ordering_to_compatible = train_args.ordering_to_compatible
    tau_end=torch.tensor(train_args.tau_end, device=device)
    tau_start=torch.tensor(train_args.tau_start, device=device)
    tau_decay = train_args.tau_decay
    torch.manual_seed(train_args.random_seed)   # set the random seed

    torch.set_printoptions(threshold=1000, precision=3, sci_mode=False) 

    assert tau_start!=0 and tau_end!=0 and epoch_num!=0
 

    train_dataset = AssignmentDataset(train_file, device)




    original_train_dataset = copy.deepcopy(train_dataset)
    original_epoch_num = epoch_num
    bestModel_of_firstIteration = None

    BDD_s, apply_s, bdd_roots = [], [], []
    prediction_to_refine, op_for_refine = None, None
    bdd, bdd_root = None, None

    stime = time.time()
    # timeout_per_iter = int(timeout/iterations)  
    tau_decay_cur_iter = tau_decay


    for iter_id in range(iterations):

        stime_cur_iter = time.time()
        

        print(f'\n--- {iter_id}th-iteration training ---')
        if iter_id!=0:
            epoch_num = original_epoch_num
            tau_decay_cur_iter = None
            print(f"prediction_to_refine: {prediction_to_refine} | op_for_refine: {op_for_refine} | ordering_to_compatible: {ordering_to_compatible}")
        else:
            pass




        dataset_size, atom_size, pos_distribution, minority_class = util.dataset_statistic(train_dataset)

        assert net_depth>=3 and net_depth<=atom_size and net_width>=2


        # if pos_account>0.75 or pos_account<0.25:        
        if False:
            class_weights = [(1-pos_account) if label else pos_account for data, label in train_dataset]
            sampler = WeightedRandomSampler(class_weights, len(train_dataset), replacement=True)    
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            print('use WeightedRandomSampler to balance the traning dataset...')
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        print('\n>training...\t', train_file)
        print('Class Distribution:  %.3f'%(pos_distribution))
        print(f"dataset_size: {dataset_size} | feature_cnt: {atom_size} | minority_class: {minority_class} | batch_size: {batch_size}")
        print('net_depth: %d | net_width: %d'%(net_depth, net_width))



        if atoms_chosen != None:  

            atoms_chosen = util.list_str_to_indices(atoms_chosen, device=device)
   
            assert len(atoms_chosen)<=atom_size and max(atoms_chosen)<atom_size and  min(atoms_chosen)>=0    
            
            


        feature_id_ordering = None
        if ordering_to_compatible is not None:
            feature_id_ordering = util.list_str_to_indices(ordering_to_compatible, device=device)

        model = OBDDNet(net_width, net_depth, atom_size, atoms_chosen=atoms_chosen, tau_start=train_args.tau_start)
        model.to(device)
        # model.train()        
        torch.save(model.state_dict(), model_dir)   


        if LOG:
            # tensorboard --logdir=filedir, then click the "SCALARS" in the TensorBoard plane
            writer = SummaryWriter(log_file + TIMESTAMP) 
            # writer = SummaryWriter(log_file) 



        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)     
    

        best_F1_score = -1
        best_train_acc = -1

        epoch0_stime = time.time()
        epoch_num_eval = epoch_num 

        timeout_left = timeout-time.time()+stime

        timeout_per_iter = int(timeout_left/(iterations-iter_id))    

        print(f"timeout left... {timeout_left}s")

        for epoch in range(epoch_num+1):

            epoch_loss = 0
            
            for data_x, data_y in train_dataloader:

                if (time.time()-stime_cur_iter > timeout_per_iter) or (time.time()-stime > timeout):            
                    break 

                prediction, _ = model(data_x)  # forward

                if prediction_to_refine is not None:
                    raw_loss = (prediction[:,0] - data_y)**2
                    weight_vec = data_y.clone().detach()
                    # a5 = 2.5
                    if prediction_to_refine==0:
                        weight_vec *= train_args.a5       
                        weight_vec += 1       
                    elif prediction_to_refine==1:
                        weight_vec = 1 - weight_vec
                        weight_vec *= train_args.a5
                        weight_vec += 1
                    else:
                        raise ValueError()
                    weighted_loss = raw_loss.mul(weight_vec)
                    loss = torch.mean(weighted_loss)

                else:
                    loss = torch.mean((prediction[:,0] - data_y)**2)

                loss += train_args.a1 * torch.sum(torch.relu(torch.sum(model.enc_dec, dim=0) - 1))      
                



                loss += train_args.a2 * (- torch.sum(model.enc_dec.mul(torch.log(model.enc_dec + 1e-5)))) 
                loss += train_args.a3 * (- torch.sum(model.enc_left.mul(torch.log(model.enc_left + 1e-5)))) 
                loss += train_args.a3 * (- torch.sum(model.enc_right.mul(torch.log(model.enc_right + 1e-5)))) 


                loss += train_args.a4 * (torch.sum(model.enc_left**2) + torch.sum(model.enc_right**2)) 


                if ordering_to_compatible is not None:
                    level_ord_rel = util.closure_relation_of_ordering(model.net_depth, device=device)
                    feature_ord_rel__forbid = 1 - util.closure_relation_of_ordering(len(feature_id_ordering), device=device)
                    sub_enc_dec =  model.enc_dec.matmul( util.columns_selector(model.atom_size, feature_id_ordering, device=device))
                    feature_ord_rel = sub_enc_dec.t().matmul(level_ord_rel).matmul(sub_enc_dec)
                    loss += train_args.a6 * torch.mean(feature_ord_rel.mul(feature_ord_rel__forbid)) 

                epoch_loss += loss.detach().cpu().numpy()


                # zero_grad
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()


                if (time.time()-stime_cur_iter > timeout_per_iter) or (time.time()-stime > timeout):
                    break 


            torch.set_printoptions(threshold=1000, precision=3, sci_mode=False) 



            with torch.no_grad():
                model.eval()
                train_acc, F1_macro, F1_score, precision, recall, is_ordered, nodes_num = model.score(train_dataset, model_type='interpreted', average=None) 
                model.train()


            if train_args.opt_metrics == 'F1':
                if F1_macro > best_F1_score : 
                    best_F1_score = F1_macro
                    torch.save(model.state_dict(), model_dir)

                if F1_macro + train_args.backtrack_dF1 < best_F1_score:
                    model.load_state_dict(torch.load(model_dir))

            if train_args.opt_metrics == 'acc':
                if train_acc > best_train_acc: 
                    best_train_acc = train_acc
                    torch.save(model.state_dict(), model_dir)
                
                if train_acc + train_args.backtrack_dacc < best_train_acc:
                    model.load_state_dict(torch.load(model_dir))



            if epoch==2:                                    
                epoch_cnt = epoch + 1                       
                time_per_epoch = (time.time() - epoch0_stime) / epoch_cnt
                if tau_decay_cur_iter is None:    
                    epoch_num_eval = int(timeout_per_iter / time_per_epoch)
                    print("evaluating (max) epoch numbers can be trained in the time limit ...\n", \
                    f"\t>preset timeout_per_iteration: {timeout_per_iter}s , estimated time_per_epoch: {time_per_epoch}s, estimated epoch_num: {epoch_num_eval}")
                    if epoch_num <= epoch_num_eval:
                    
                        tau_decay_cur_iter = round(pow(float(tau_end/tau_start), 2/(epoch_num-epoch_cnt)), 3) if epoch_num>epoch_cnt else 0

                    else:
                        tau_decay_cur_iter = round(pow(float(tau_end/tau_start), 2/(epoch_num_eval-epoch_cnt)), 3) if epoch_num_eval>epoch_cnt else 0


            model.tau.data = max(tau_end, model.tau.data*tau_decay_cur_iter) if tau_decay_cur_iter is not None else tau_end     

            if epoch % 20==0:
                print('epoch: %d | time: %.3f | loss: %.5f | epoch_acc: %.3f | epoch_F1_macro: %.3f | is_ordered: %s | nodes_num: %d| lr: %.5f'%(epoch, time.time()-stime_cur_iter, epoch_loss, \
                    train_acc, F1_macro, str(is_ordered), nodes_num, optimizer.state_dict()['param_groups'][0]['lr']))



            if (time.time()-stime_cur_iter > timeout_per_iter) or (time.time()-stime > timeout):
                # print('Timeout!!\n epoch: %d | time: %.3f | loss: %.5f | epoch_acc: %.3f | epoch_F1: %.3f | epoch_prec: %.3f | epoch_recall: %.3f | is_ordered: %s | nodes_num: %d| lr: %.5f'%(epoch, time.time()-stime_cur_iter, epoch_loss, \
                #     train_acc, F1_score[minority_class], precision[minority_class], recall[minority_class], str(is_ordered), nodes_num, optimizer.state_dict()['param_groups'][0]['lr']))
                print('epoch: %d | time: %.3f | loss: %.5f | epoch_acc: %.3f | epoch_F1_macro: %.3f | is_ordered: %s | nodes_num: %d| lr: %.5f'%(epoch, time.time()-stime_cur_iter, epoch_loss, \
                    train_acc, F1_macro, str(is_ordered), nodes_num, optimizer.state_dict()['param_groups'][0]['lr']))
                break 
            
            if epoch_loss < min_loss:     
                break 

            if LOG:
                # save the log
                writer.add_scalar("loss-epoch.", epoch_loss, epoch) 
                writer.add_scalar("train_F1_macro-epoch.", F1_macro, epoch) 
                writer.add_scalar("best_train_F1_macro-epoch.", best_F1_score, epoch) 

                cur_time = time.time()-stime
                writer.add_scalar("loss-time.", epoch_loss, cur_time) 
                writer.add_scalar("train_F1_macro-time.", F1_macro, cur_time) 
                writer.add_scalar("best_train_F1_macro-time.", best_F1_score, cur_time) 

                writer.add_scalar("time-epoch.", cur_time, epoch) 

        train_time = time.time()-stime

        if LOG:
            # writer.add_graph(model, data_x)    
            writer.close()

        '''Final evaluation for both network model and the interpreted BDD model'''
        # At last, get the interpreted OBDD of the saved model and evaluate the training set 
        with torch.no_grad():
            model_params = torch.load(model_dir)
            model.load_state_dict(model_params)
            model.eval()
            
            # net_train_acc, net_train_F1_macro, net_train_F1, net_train_prec, net_train_rec = model.score(train_dataset, model_type='train-original', average=None) 
            # print(f"For train-original model: \n net_train_acc: {net_train_acc} | net_train_F1_macro：{net_train_F1_macro} | net_train_F1: {net_train_F1} | net_train_prec:{net_train_prec} | net_train_rec: {net_train_rec}")
            net_train_acc, net_train_F1_macro, net_train_F1, net_train_prec, net_train_rec = model.score(train_dataset, model_type='test-original', average=None) 
            print(f"For test-original model:  \n net_train_acc: {net_train_acc} | net_train_F1_macro：{net_train_F1_macro} | net_train_F1: {net_train_F1} | net_train_prec:{net_train_prec} | net_train_rec: {net_train_rec}")
            
            stime_model_score = time.time()
            train_acc, train_F1_macro, train_F1, train_prec, train_rec, is_ordered, nodes_num = model.score(train_dataset, model_type='interpreted', average=None) 
            print(f"model_score uses time : {time.time()-stime_model_score}s")    
            
            print(f"For interpreted model: \n train_acc: {train_acc} | train_F1_macro: {train_F1_macro} | train_F1: {train_F1} | train_prec:{train_prec} | train_rec: {train_rec} | is_ordered: {is_ordered} | nodes_num (for BDD): {nodes_num}")
    
        
            BDD = model.decode(atoms_name=train_dataset.get_atom_description())       
            feature_ordering_,_ = BDD.get_feature_ordering()
            print(f"The feature ordering of the learned BDD: {feature_ordering_}")


            if iter_id==0:
                bestModel_of_firstIteration = model_params
                bdd, bdd_root, _ = BDD.construct_bdd()  # to get the reduced OBDD (but it is not in the simple OBDD form, containing complement edges) 

                bdd_actual_size = util.bdd_actual_size(bdd, bdd_root) # to get the reduced OBDD in the simple OBDD form as defined in the paper. NOTE: it do not contain complement edges any more
                BDD_size = BDD.get_vertices_cnt()
                print(f"BDD_size (for non-reduced OBDD): {BDD_size} vs. bdd_actual_size (for reduced OBDD): {bdd_actual_size}")
                dot = BDD.get_dot_description() 
                with open("learned_OBDD.dot", "w") as fo:
                    fo.write(dot)

                # bdd.incref(bdd_root)    #  to make node bdd_root to persist after garbage collection, it needs to be actively referenced at least once
            else:
                stime_bdd_op = time.time()
                bdd, bdd_root = BDD.bdd_op_BDD(bdd, bdd_root, BDD, apply=op_for_refine)
                # bdd.incref(bdd_root)    #  to make node bdd_root to persist after garbage collection, it needs to be actively referenced at least once
                print(f"bdd_op_BDD uses time : {time.time()-stime_bdd_op}s")    


            BDD_s.append(BDD)
            bdd_roots.append(bdd_root)

            if (time.time()-stime > timeout):            
                train_time = time.time()-stime
                break 

            if iter_id<iterations-1:
                stime_refine_guide = time.time()
                train_dataset, prediction_to_refine, op_for_refine, ordering_to_compatible = refine_guide(original_train_dataset, bdd, bdd_root)
                print(f"refine_guide uses time : {time.time()-stime_refine_guide}s")   
                apply_s.append(op_for_refine)

                if (time.time()-stime > timeout):            
                    break

        

    dump_BDD_s(BDD_s)
    print("apply_s: " + ', '.join(apply_s))
    bdd_s, bdd_root_s, bdd_best, bdd_root_best = print_bdds_info(original_train_dataset, bdd, bdd_roots, train_args.opt_metrics)
    nodes_num = util.bdd_actual_size(bdd_best, bdd_root_best)
    _, train_acc, train_F1_macro, train_F1, train_prec, train_rec = util.bdd_predict_score(original_train_dataset, bdd_best, bdd_root_best)
    '''
    NOTE: In this version, we save only the trained network model in the first iteration.
    To get the performation of final learned bdd, please evaluate on "the bdd with root bdd_roots[-1]" or "the merged bdd according to BDD_s and apply_s".
    '''
    torch.save(bestModel_of_firstIteration, model_dir)   

    print("train_time :", train_time)   

    return train_time, net_train_acc, net_train_F1_macro, net_train_F1, net_train_prec, net_train_rec, \
    train_acc, train_F1_macro, train_F1, train_prec, train_rec, is_ordered, nodes_num, \
    BDD_s, apply_s, bdd_s, bdd_root_s, bdd_best, bdd_root_best





def get_train_argument(args_list=None):
    train_parser = argparse.ArgumentParser(description='Main script for train')
    train_parser.add_argument('--train_file', type=str, required=True)
    train_parser.add_argument('--model_file', type=str, required=False, default='model/train_model')
    train_parser.add_argument('--log_file', type=str, required=False, default='log/train_log')

    train_parser.add_argument('--net_depth', type=int, required=False, default=6, help='(maximal) number of decision levels of the learned OBDD')
    train_parser.add_argument('--net_width', type=int, required=False, default=32, help='(maximal) number of nodes in each decision level. Default value:  2^(net_depth)-1')
    train_parser.add_argument('--epoch', type=int, required=False, default=3000, help='epoch number')
    train_parser.add_argument('--lr', type=float, required=False, default=0.2, help='learn rate')
    train_parser.add_argument('--iterations', type=int, required=False, default=1, help='the number of iterations (during iterative learning)')
    train_parser.add_argument('--a1', type=float, required=False, default=1, help='regular cof. towards read-once poperty')
    train_parser.add_argument('--a2', type=float, required=False, default=1e-6, help='regular cof. towards 0-1-binarization for params enc_dec')
    train_parser.add_argument('--a3', type=float, required=False, default=1e-6, help='regular cof. towards 0-1-binarization for params enc_lef & enc_right')
    train_parser.add_argument('--a4', type=float, required=False, default=0, help='regular cof. towards size optimization')
    train_parser.add_argument('--a5', type=float, required=False, default=1.1, help='regular cof. towards the loss weight of the opposite class of the prediction to be refined during iterative learning')
    train_parser.add_argument('--a6', type=float, required=False, default=0, help='regular cof. towards persisting the compatibility with the given ordering: ordering_to_compatible during iterative learning')
    train_parser.add_argument('--batch_size', type=int, required=False, default=
    512, help='batch size')
    train_parser.add_argument('--random_seed', type=int, required=False, default=2024, help='random_seed (for model initalization)')
    train_parser.add_argument('--threshold', type=float, required=False, default=0.5, help='threshold for classification')
    train_parser.add_argument('--atoms_chosen', type=str, required=False, default=None, help='specify the indices of atoms chosen. e.g, \'0, 2, 5, 3\'')
    train_parser.add_argument('--ordering_to_compatible', type=str, required=False, default=None, help='specify the atoms ordering that the trained classifier are expected be compatible with. e.g, \'0, 2, 5, 3\'')
    train_parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device')

    train_parser.add_argument("--opt_metrics", choices=["acc", "F1"], default="acc", help="specify the optimization metrics used to select model during training: acc or (macro) F1")
    train_parser.add_argument('--timeout', type=int, default=900, help='timeout for training process (unit: s)')
    train_parser.add_argument('--tau_start', type=float, default=10, help='the initial value of the temperature parameter tau in Gumbel Softmax function')
    train_parser.add_argument('--tau_end', type=float, default=1, help='the terminal value of the temperature parameter tau')
    train_parser.add_argument('--tau_decay', type=float, default=None, help='the decay factor of thetemperature parameter tau. if None, an proper value will be automatically taken based on epoch, timeout and iterations')
    train_parser.add_argument('--backtrack_dF1', type=float, default=0.3, help='the (macro) F1 performance degradation threshold to trigger backtracking to the acquired best model')
    train_parser.add_argument('--backtrack_dacc', type=float, default=0.3, help='the acc performance degradation threshold to trigger backtracking to the acquired best model')

    
    if args_list==None:
        train_args = train_parser.parse_args()
    else:
        train_args = train_parser.parse_args(args_list)
    return train_args






