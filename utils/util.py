import torch
from torch import nn
from torch.nn.parameter import Parameter

import dd.autoref as _bdd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import json
import os


def features_to_indices(feature_names, feature_name_ordering):

    feature_id_ordering = []
    for feature_name in feature_name_ordering:
        feature_id = feature_names.index(feature_name)
        feature_id_ordering.append(feature_id)

    return feature_id_ordering




def list_str_to_indices(list_str, device):

    list_indices = []
    for elem in list_str.split(','):
        elem = elem.strip()
        if elem!='':
            list_indices.append(int(elem))

    list_indices = torch.tensor(list_indices, dtype=torch.float32, requires_grad=False).to(device)

    return list_indices




def columns_selector(matrix_width, feature_id_ordering, device):
    '''
    @param matrix_width: the width of a matrix to select columns
    @param feature_id_ordering: columns to be selected
    
    @return selector_matrix: a 2-dim tensor that can used to select columns from matrix via  matrix \matmul selector_matrix
    '''
    selector_matrix = []
    # columns_num = len(matrix[0])
    columns_num = matrix_width
    for feature_id in feature_id_ordering:
        onehot_vector = [0] * columns_num
        onehot_vector[int(feature_id)] = 1
        selector_matrix.append(onehot_vector)
    selector_matrix = torch.tensor(selector_matrix, dtype=torch.float32, requires_grad=False).to(device)
    selector_matrix = torch.transpose(selector_matrix, 0, 1)
    # selector_matrix = selector_matrix.t()
    return selector_matrix




def closure_relation_of_ordering(ordering_len, device):
    '''
    @param ordering_len: the len of a given ordering [f1, ..., f_len]
    @return closure_relation: a two-dim tensor representing the closure of the relation of the ordering
    '''
    # dim = len(ordering)
    dim = ordering_len
    closure_relation = [[(1 if row<col else 0) for col in range(dim)] for row in range(dim)]
    closure_relation = torch.tensor(closure_relation, dtype=torch.float32, requires_grad=False).to(device)
    return closure_relation







def dataset_statistic(dataset):
    '''
    @param dataset: a AssignmentDataset object
    '''
    dataset_size = dataset.size()
    feature_cnt = dataset.get_atom_size()
    pos_distribution = dataset.get_class_distribution()
    minority_class = 1 if pos_distribution<=0.5 else 0
    # print('Class Distribution:  %.3f'%(pos_distribution))
    return dataset_size, feature_cnt, pos_distribution, minority_class




# ======================================

def get_bdd_feature_ordering(bdd):
    feature_name_s = list(bdd.vars.keys())    # each element is "f{feature_id}" where feature_id in feature_ordering
    feature_ordering = [int(feature_name[1:]) for feature_name in feature_name_s]
    return feature_ordering     # the whole features declared in the bdd

def get_bdd_feature_ordering__support(bdd, bdd_root):
    feature_name_s = list(bdd.vars.keys())  # the whole features declared in the bdd (regardless of these features are mentioned in the (sub)-bdd root at bdd_root)
    support_name_s = bdd_root.support       # the features mentioned in the (sub)-bdd root at bdd_root NOTE: just a set but not a list
    feature_ordering = [int(feature_name[1:]) for feature_name in feature_name_s if feature_name in support_name_s]
    return feature_ordering



def copy_bdd(bdd, bdd_root_s, bdd_root_to_persist):

    # copy to another BDD manager
    new_bdd = _bdd.BDD()
    new_bdd.declare(*bdd.vars)
    new_bdd_root = bdd.copy(bdd_root_to_persist, new_bdd)

    # nodes_num_1 = len(new_bdd)
    new_bdd.collect_garbage()  # remove unused nodes, so that len(bdd) can return exact bdd size 

    return new_bdd, new_bdd_root



def _bdd_predict(dataset, bdd, bdd_root):

    data = dataset.get_data()

    feature_name_s = list(bdd.vars.keys())    # each element is "f{feature_id}" where feature_id in feature_ordering
    # print(feature_name_s)
    feature_ordering = [int(feature_name[1:]) for feature_name in feature_name_s]
    
    prediction = []
    for exam in data:
    # for idx, exam in enumerate(data):
    # for exam in data[:2]:
        assignment = {}
        for feature_id in feature_ordering:

            ith = feature_ordering.index(feature_id)

            if int(exam[feature_id]) == 1:
                assignment[feature_name_s[ith]] = True
            elif int(exam[feature_id]) == 0:
                assignment[feature_name_s[ith]] = False
            else:
                raise ValueError()


        # substitute constants for variables (cofactor)
        res = bdd.let(assignment, bdd_root)

        pred = int(res != bdd.false)
        prediction.append(pred)

        # if idx%2000==0: 
        #     bdd.collect_garbage() # do not use here     
        #     print(f" the number of nodes in bdd manager:{len(bdd)}")


    # bdd.collect_garbage() # do not use here     # 
    # print(f" the number of nodes in bdd manager:{len(bdd)}")

    # print(prediction)
    return prediction



def bdd_predict_score(dataset, bdd, bdd_root):
    
    prediction = _bdd_predict(dataset, bdd, bdd_root)

    label = dataset.labels.cpu().numpy().tolist()
    acc = accuracy_score(label, prediction)
    f1 = f1_score(label, prediction, average=None)
    f1_macro = f1_score(label, prediction, average="macro")
    prec = precision_score(label, prediction, average=None, zero_division=0)
    rec = recall_score(label, prediction,  average=None)

    # print(f"acc {acc} | f1 {f1} | prec {prec} | recall {rec}")
    return prediction, acc, f1_macro, f1, prec, rec


def bdd_dump_str(bdd, bdd_root):
    jsonfile_temp = "bdd_temp.json"
    bdd.dump(jsonfile_temp, roots=[bdd_root])
    bdd_str = ""
    with open(jsonfile_temp, "r") as f:
        bdd_str = f.read()
        # print(bdd_str)

    # if os.path.exists(jsonfile_temp):
    #     os.remove(jsonfile_temp)
    return bdd_str


def _find_key_by_value(mydict, target_value):
    for key, value in mydict.items():
        if value == target_value:
            return key
    return None  

def _modify_key(key):
    return str(key).replace("-", "_")


def _get_neg_node_id(label):
    if label=="T":
        return "F"
    elif label=="F":
        return "T"
    else:
        return -label



def _depict_BDD_from_bdd(bdd_dict, BDD_file="BDD.dot"):

    with open(BDD_file, "w") as fo:
        fo.write("digraph G {\n")

        level_of_var =  bdd_dict["level_of_var"]

        for key in bdd_dict.keys():

            if key=="level_of_var" :
                continue
            elif key=="roots":
                continue
            elif key=="F" or key=="T":
                fo.write("\tn{}[label=\"{}\", shape=square]\n".format(key, key))
            else:
                var = _find_key_by_value(level_of_var, bdd_dict[key][0])
                left_node_id = bdd_dict[key][2]         # NOTE
                right_node_id = bdd_dict[key][1]        # NOTE

                fo.write("\tn{}[label=\"{}\"]\n".format(_modify_key(key), var))
                fo.write("\tn{} -> n{};\n".format(_modify_key(key), _modify_key(left_node_id)))
                fo.write("\tn{} -> n{}[style=\"dotted\"];\n".format(_modify_key(key), _modify_key(right_node_id)))
        fo.write("}")



def bdd_recover(bdd_str):
    # jsonfile_temp=bdd_file
    # bdd_str = get_bdd_dump_str(jsonfile_temp=jsonfile_temp)
    bdd_dict = json.loads(bdd_str)

    bdd_recover_dict = dict()       # recover some hidden nodes accroding to the complement edge

    level_of_var = bdd_dict["level_of_var"]
    to_visit_ids = set(bdd_dict["roots"])       # Set[int/str]
    visited_ids = set()     # Set[int/str]

    to_visit_ids_copy = to_visit_ids.copy()

    bdd_recover_dict["level_of_var"] = level_of_var
    bdd_recover_dict["roots"] = bdd_dict["roots"]
    
    while len(to_visit_ids_copy)!=0:

        for node_id in to_visit_ids_copy:

            # print("visited_ids:",visited_ids)
            # print("to_visit_ids:", to_visit_ids)
            # print("-----")

            node_id_str = str(node_id)  

            if node_id_str=="T" or node_id_str=="F":
                if node_id_str not in bdd_recover_dict.keys():
                    bdd_recover_dict[node_id_str] = [node_id_str]     # "T":["T"],   "F":["F"]

            else:
                if node_id>0:
                    if node_id_str not in bdd_dict.keys():      # in this case, node_id is exactly roots, and it represents True or False
                        bdd_recover_dict["roots"] = "T"
                        ite_temp = None
                    else:
                        ite_temp = bdd_dict[node_id_str].copy()
                elif node_id<0:
                    # node_id_str = str(-node_id)
                    if str(-node_id) not in bdd_dict.keys():    # in this case, node_id is exactly roots, and it represents True or False
                        bdd_recover_dict["roots"] = "F"
                        ite_temp = None
                    else:
                        ite_temp = bdd_dict[str(-node_id)].copy()      
                        ite_temp[1] = _get_neg_node_id(ite_temp[1])
                        ite_temp[2] = _get_neg_node_id(ite_temp[2])
                    
                else:
                    raise ValueError(f"ERROR! node_id is {node_id}.")
                    # pass

                if ite_temp is not None:
                    if node_id_str not in bdd_recover_dict.keys():
                        bdd_recover_dict[node_id_str] = ite_temp

                    for ch_id in ite_temp[1:]:
                        if ch_id not in visited_ids:
                            # print("enter")
                            to_visit_ids.add(ch_id)
                            # print("to_visit_ids:", to_visit_ids)


            to_visit_ids.remove(node_id)
            visited_ids.add(node_id)
            # print("to_visit_ids:", to_visit_ids)
        
        to_visit_ids_copy = to_visit_ids.copy()
        # print("to_visit_ids_copy:", to_visit_ids_copy)

    # bdd_recover_dict["level_of_var"] = level_of_var
    # bdd_recover_dict["roots"] = bdd_dict["roots"]

    # print("visited_ids: ", visited_ids)
    # print("bdd_recover_dict: ", bdd_recover_dict)

    bdd_actual_size = len(visited_ids)

    return bdd_recover_dict, bdd_actual_size, visited_ids



def bdd_actual_size(bdd, bdd_root):
    '''
    Compute the actual size of the bdd with root bdd_root. That is, to get an OBDD of simple form (as defined in the paper) that contains no complement edges.
    NOTE: as the implementation of bdd object use complement edges (which allows to omit the appearance negation of an appeaing node), 
    we can not get the actual number of nodes via compute "len(bdd)".
    Here we will first call the procedure "bdd_recover" to convert it into the simple OBDD form, and then compute its size.  
    '''

    bdd.collect_garbage() 
    # bdd_size = len(bdd)

    bdd_str = bdd_dump_str(bdd, bdd_root)
  
    bdd_recover_dict, bdd_actual_size, _ = bdd_recover(bdd_str)

    return bdd_actual_size


# ---

def _get_bdd_dump_str(jsonfile):
    bdd_str = ""
    with open(jsonfile, "r") as f:
        bdd_str = f.read()
    return bdd_str


def _get_mapping_of_feat_ids_2_feat_names(ids_names_file):
    '''
        :param ids_names_file: a json file in the following form: 
            {
            "feat_ids": [3, 1, 9, 18],
            "feat_names": ["feat_name_1", "feat_name_2", "feat_name_3", "feat_name_4"]
            }
    '''
    with open(ids_names_file, 'r') as file:
        _dict = json.load(file)

    assert "feat_ids" in _dict.keys()
    assert "feat_names" in _dict.keys()
    feat_ids = _dict["feat_ids"]
    feat_names = _dict["feat_names"]
    assert len(feat_ids)==len(feat_names)

    feat_ids_2_feat_names = dict()
    for i, feat_id in enumerate(feat_ids):
        feat_ids_2_feat_names[f"f{feat_id}"] = feat_names[i]

    return feat_ids_2_feat_names



def depict_BDD_from_bddfile(bdd_jsonfile, feat_ids_2_feat_names_jsonfile=None):
    bdd_str = _get_bdd_dump_str(bdd_jsonfile)
    bdd_recover_dict, _, _ = bdd_recover(bdd_str)

    if feat_ids_2_feat_names_jsonfile:
        feat_ids_2_feat_names = _get_mapping_of_feat_ids_2_feat_names(feat_ids_2_feat_names_jsonfile)
        level_of_var = bdd_recover_dict["level_of_var"]
        new_level_of_var = {feat_ids_2_feat_names.get(k, k): v for k, v in level_of_var.items()}  # replace the feature id to its corresponding feature name
        bdd_recover_dict["level_of_var"] = new_level_of_var

    print(f"depict BDD from {bdd_jsonfile}...\t")
    _depict_BDD_from_bdd(bdd_recover_dict)
    print(f"Done")


    
if __name__ == '__main__':

    bdd_jsonfile = "to_depict_recovered_bdd.json"
    feat_ids_2_feat_names_jsonfile = "feat_ids_2_feat_names.json"

    depict_BDD_from_bddfile(bdd_jsonfile, feat_ids_2_feat_names_jsonfile)