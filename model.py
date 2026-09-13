import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import time

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data import AssignmentDataset
from utils.bdd import BinaryDecisionDiagram
# import myconfig



class CenteredLayer(torch.nn.Module):
    '''
        Class CenteredLayer. apply the minmax operation (a variant of relu) to a tensor

        :param x: a tensor
        :return x: a new tensor applied the minmax operation
    '''
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    
    def forward(self, x):
        x[x<0] = x[x<0]*0.01        # for an element<0  
        x[x>1] = x[x>1]*0.01 + 0.99 # for an element>1 
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output



class OBDDNet(torch.nn.Module):
    '''
        Class OBDD-Net.

        :param XX: 
    '''
    def __init__(self, net_width, net_depth, atom_size, atoms_chosen=None, tau_start=1):
        super(OBDDNet, self).__init__()
        self.net_width = net_width
        self.net_depth = net_depth
        self.atom_size = atom_size
        self.atoms_chosen = atoms_chosen

        self.myrelu = CenteredLayer()
        self.tau=Parameter(torch.tensor(tau_start, dtype=torch.float32), requires_grad=False)

       
        ''' define trainable parameters'''

        self.params_left  = torch.nn.ParameterList([Parameter(torch.randn(net_width if level else 1, (net_depth-1-level)*net_width + 2)) for level in range(net_depth)]) # list(net_depth, ). root_(1), mid_(net_width), leaf_(2) 
        self.params_right = torch.nn.ParameterList([Parameter(torch.randn(net_width if level else 1, (net_depth-1-level)*net_width + 2)) for level in range(net_depth)]) # root_(1), mid_(net_width), leaf_(2)

        
        for x in self.params_left:
            nn.init.orthogonal_(x)
        for x in self.params_right: 
            nn.init.orthogonal_(x)

     

        if self.atoms_chosen != None:
            self.params_dec  =  Parameter( torch.randn(net_depth, atom_size) )   # a mapping from atoms to decision vars
            mask = torch.ones_like(self.params_dec)
            mask[:, self.atoms_chosen.long()] = 0
            with torch.no_grad():
                self.params_dec[mask.bool()] = -1000
        else:
           
            self.params_dec  = Parameter( torch.randn(net_depth, atom_size) )   
            nn.init.orthogonal_(self.params_dec)



        ''' some auxiliary variables '''
        self.leaf_nodes = [ (net_width * (net_depth-1)) + 1, (net_width * (net_depth-1)) + 2 ]
        # use for recording the reduced faithful OBDD encoding 
        self.interpret_left = None
        self.interpret_right = None
        self.interpret_dec = None
        self.BDD_nodes = None
        self.BDD_levels = None





    def bulid_OBDD_encoding(self, test_flag=False):
        '''
            Bulid OBDD-encoding \Enc from OBDD-Net \Para via softmax operation. 
        '''
        device=self.params_left[0].device
        net_width = self.net_width
        net_depth = self.net_depth
        
        # the completement of the params_left/params_right (to form the upper (non-leaf) part  of transfer_martrix)
        params_padding = [torch.full((net_width if level else 1, 1 + level*net_width), -1000, dtype=torch.float32).to(device) for level in range(net_depth)] 
        # the lower (leaf) part of transfer_martrix
        bottom_padding = torch.zeros(2, 1 + (net_depth-1)*net_width + 2).to(device) 
        bottom_padding[0,-2] = 1  # for convenience to design loss, we redefine the Tleaf as a terminal node with a self-loop 
        bottom_padding[1,-1] = 1  # for convenience to design loss, we redefine the Fleaf as a terminal node with a self-loop 

        left_matrix = [torch.cat((params_padding[level], self.params_left[level]), dim=1) for level in range(net_depth)]      # the upper part of transfer_left_martrix
        left_matrix = torch.cat(left_matrix, dim=0)   
        right_matrix = [torch.cat((params_padding[level], self.params_right[level]), dim=1) for level in range(net_depth)]    # the upper part of transfer_right_martrix
        right_matrix = torch.cat(right_matrix, dim=0)   

        # NOTE: transfer_matrix_size = 1 + (net_depth-1)*net_width + 2
        if test_flag:
            self.enc_left = torch.cat((torch.softmax(left_matrix, dim=1), bottom_padding), dim=0)        # (dim: transfer_matrix_size * transfer_matrix_size). corresponds to the transfer_left_martrix
            self.enc_right = torch.cat((torch.softmax(right_matrix, dim=1), bottom_padding), dim=0)      # (dim: transfer_matrix_size * transfer_matrix_size). corresponds the transfer_left_martrix
            self.enc_dec = torch.softmax(self.params_dec, dim=1)
        else:
            self.enc_left = torch.cat((F.gumbel_softmax(left_matrix, tau=self.tau, hard=False, dim=1), bottom_padding), dim=0)
            self.enc_right = torch.cat((F.gumbel_softmax(right_matrix, tau=self.tau, hard=False, dim=1), bottom_padding), dim=0)
            self.enc_dec = F.gumbel_softmax(self.params_dec, tau=self.tau, hard=False, dim=1)

            # # NOTE: use this for ablation study on faithfulness
            # self.enc_left = torch.cat((torch.sigmoid(left_matrix), bottom_padding), dim=0)
            # self.enc_right = torch.cat((torch.sigmoid(right_matrix), bottom_padding), dim=0)
            # self.enc_dec = torch.sigmoid(self.params_dec)

        # print('enc_left', self.enc_left)      
        # print('enc_right', self.enc_right)      
        # print('enc_dec', self.enc_dec)      


    def forward(self, data_x, REFRESH_ENCODING=True):
        '''
            Forward.  
        '''
        device=data_x.device
        if REFRESH_ENCODING:
            self.bulid_OBDD_encoding()
        batch_size, _ = data_x.size() 

        net_width = self.net_width
        net_depth = self.net_depth


        data_x_selected = self.enc_dec.unsqueeze(0).matmul(data_x.unsqueeze(2))  # (dim: batch_size * net_depth * 1). select and reorder the data_x

        filter_left_mid = data_x_selected[:,1:,:].repeat(1, 1, net_width).to(device)    # (dim: batch_size * net_depth-1 * net_width).  the probability to moving left branch of each interal node, under an example


        filter_left = torch.cat((
                data_x_selected[:,0:1], \
                filter_left_mid.view(batch_size, (net_depth-1) * net_width, 1), \
                torch.ones(batch_size, 2, 1).to(device) \
            ), dim=1)      # (dim: batch_size * transfer_matrix_size * 1).
        filter_right = 1 - filter_left


        transfer_matrix = self.enc_left.mul(filter_left) + self.enc_right.mul(filter_right)  # (dim: batch_size * transfer_matrix_size * transfer_matrix_size)


        root_reach = transfer_matrix[:,0:1,:]   # the (1-step) reachability of the root node


        root_reach_matrix = root_reach          # restore the the (i-step) reachability info of the root node for 1 \leq i \leq net_depth
        for step in range(1, net_depth+1):      # the (i-step) reachability of the root node
            if step==1:
                next_root_reach = root_reach
            else:
                cur_root_reach = next_root_reach 
                next_root_reach = self.myrelu(cur_root_reach.matmul(transfer_matrix)) 

                root_reach_matrix = torch.cat((root_reach_matrix, next_root_reach), dim=1) 

        result = root_reach_matrix[:,-1,-2:]   

        return result, root_reach_matrix


    # def interpret_BDD(self, SAVE_DOT=False, file_name=None, atoms_name=None):
    def interpret_BDD_v1(self, SAVE_DOT=False, file_name=None, atoms_name=None):
        '''
        filter the useful information for interpreting candidate OBDD from the learning connection information

        '''
        self.bulid_OBDD_encoding(test_flag=True)
        self.is_ordered = True

        with torch.no_grad():
            '''interpretation'''    # Binarization: reset the maximal element  1 in dim1, and 0s the others 
          
            interpret_left = torch.zeros_like(self.enc_left)
            interpret_left[torch.arange(len(interpret_left)), torch.argmax(self.enc_left, dim=1)] = 1

            interpret_right = torch.zeros_like(self.enc_right)
            interpret_right[torch.arange(len(interpret_right)), torch.argmax(self.enc_right, dim=1)] = 1

            interpret_dec = torch.zeros_like(self.enc_dec)
            interpret_dec[torch.arange(len(interpret_dec)), torch.argmax(self.enc_dec, dim=1)] = 1

            # >> filter the nodes & levels that are indeed reached from the root node
            interpret_transition = interpret_left + interpret_right
            candidate_edges = torch.nonzero(interpret_transition, as_tuple = False).tolist()  # the ordered candidate edges obtained from the interpreted transition matrix

            target_nodes = set()   
            target_nodes.add(0)  # set the root node (i.e., the 0-th node) as a target node
            for node_pair in candidate_edges:
                # add the end-point of the edge as a new target node, if its sourse can reached from the root node but is not a leaf node (so that remove its loop) 
                if (node_pair[0] in target_nodes) and (node_pair[0] < len(interpret_transition)-2): 
                    target_nodes.add(node_pair[1])  # add the target node
            
            BDD_nodes =  sorted(target_nodes)       # get all the ordered nodes that are indeed reached from the root node
            leaf_nodes = { (self.net_width * (self.net_depth-1)) + 1, (self.net_width * (self.net_depth-1)) + 2 }
            BDD_levels = sorted(set([ (int((inner_node-1)/self.net_width+1) if inner_node else 0)  for inner_node in list(set(BDD_nodes) - leaf_nodes) ])) # get the ordered levels contained in BDD
 

            # >> filter out the unused nodes & levels 
            for idx in range(len(interpret_transition)):
                if not (idx in BDD_nodes):
                    interpret_left[idx] = 0
                    interpret_right[idx] = 0

            for idx in range(self.net_depth):
                if not (idx in BDD_levels):
                    interpret_dec[idx] = 0

            interpret_transition = interpret_left + interpret_right


            '''verify the ordering property''' # sound but not complete(when not every level occur in the BDD) [Fixed. we have filtered out the unused nodes & levels]
            # check if each col of enc_dec has at most one element 1
            self.is_ordered = torch.all(torch.sum(interpret_dec, dim=0) <= 1).item() 

            
            '''BDD Visualization''' 
            if SAVE_DOT:
                with open(file_name, "w") as file:
                    file.write("digraph G {\n")
                    # file.write("\tlabel=\"{}\"\n".format(file_name))
            
                    # interpret_transition = interpret_left + interpret_right
                    todepict_edges = torch.nonzero(interpret_transition[:-2,:], as_tuple = False).tolist() # depict all the edges except that the ones starting from leaves

                    for node_pair in todepict_edges:
                        if interpret_left[node_pair[0], node_pair[1]]:
                            file.write("\tn{} -> n{};\n".format(node_pair[0], node_pair[1]))

                        if interpret_right[node_pair[0], node_pair[1]]:
                            file.write("\tn{} -> n{}[style=\"dotted\"];\n".format(node_pair[0], node_pair[1]))

                    dec_map = interpret_dec.argmax(dim=1) # 【Add】calculate the mapping from the levels to dec var [the unused level may has wrong map]
                    # target_nodes = list(target_nodes).sort()
                    for idx, node in enumerate(BDD_nodes):
                        if node==(self.net_width * (self.net_depth-1)) + 1:
                            file.write("\tn{}[label=\"{}\", xlabel=\"n{}\", shape=square]\n".format(node, "⊤", idx))
                        elif node==(self.net_width * (self.net_depth-1)) + 2: 
                            file.write("\tn{}[label=\"{}\", xlabel=\"n{}\", shape=square]\n".format(node, "⊥", idx))
                        else:   # declaration of the decision nodes
                            if atoms_name is None:
                                # node 0 is in level 0; node i>0 is in level (node-1)/self.net_width+1
                                file.write("\tn{}[label=\"x{}\", xlabel=\"n{}\"]\n".format(node, dec_map[int((node-1)/self.net_width+1) if node else 0], idx))  
                            else:
                                # use the corresponing atoms_name to denote the decision node
                                file.write("\tn{}[label=\"{}\", xlabel=\"n{}\"]\n".format(node, atoms_name[dec_map[int((node-1)/self.net_width+1) if node else 0]], idx)) 
                                # print("\tn{}[label=\"{}\", xlabel=\"n{}\"]\n".format(node, atoms_name[dec_map[int((node-1)/self.net_width+1) if node else 0]], idx)) 



                    ordering = [ (atoms_name[dec_map[level]] if atoms_name else f'x{dec_map[level]}') for level in BDD_levels]   
                    # print(ordering)     
                    file.write( "\n\tlabel=\" %s \n\t %s \"\n" % (' < '.join(ordering), file_name))

                    file.write("}")

                    
        # nodes_num = len(target_nodes)
        nodes_num = len(BDD_nodes)
        # print('BDD_nodes:,'BDD_nodes)
        self.interpret_left = interpret_left
        self.interpret_right = interpret_right
        self.interpret_dec = interpret_dec

        return self.is_ordered, nodes_num
        

    def find_argmax_not_in_set(self, vec, ids_set):
        max_val = -1000     # here we consider only the vec (1-dim-tensor) with non-negativate float vlaue
        max_val_idx = -1
        for idx in range(len(vec)):
            # if (idx not in ids_set) and (vec[idx] > max_val):
            if (vec[idx] > max_val) and (idx not in ids_set):
                max_val = vec[idx].item()
                max_val_idx = idx
        return max_val_idx, max_val



    def interpret_BDD(self, SAVE_DOT=False, file_name=None, atoms_name=None):
        '''
        filter the useful information for interpreting candidate OBDD from the learning connection information

        '''
        self.bulid_OBDD_encoding(test_flag=True)
        self.is_ordered = True

        with torch.no_grad():
            '''interpretation'''    # Binarization: reset the maximal element  1 in dim1, and 0s the others 

            # >> reset the maximal element 1 in dim1 (select the last one if it is not unique)
            # can ensure the validity of the connection
            interpret_left = torch.zeros_like(self.enc_left)
            interpret_left[torch.arange(len(interpret_left)), torch.argmax(self.enc_left, dim=1)] = 1  

            interpret_right = torch.zeros_like(self.enc_right)
            interpret_right[torch.arange(len(interpret_right)), torch.argmax(self.enc_right, dim=1)] = 1

            interpret_dec = torch.zeros_like(self.enc_dec)


            # >> filter the nodes & levels that are indeed reached from the root node
            interpret_transition = interpret_left + interpret_right
            candidate_edges = torch.nonzero(interpret_transition, as_tuple = False).tolist()  # the ordered candidate edges obtained from the interpreted transition matrix

            target_nodes = set()   
            target_nodes.add(0)  # set the root node (i.e., the 0-th node) as a target node
            for node_pair in candidate_edges:
                # add the end-point of the edge as a new target node, if its sourse can reached from the root node but is not a leaf node (so that remove its loop) 
                if (node_pair[0] in target_nodes) and (node_pair[0] < len(interpret_transition)-2): 
                    target_nodes.add(node_pair[1])  # add the target node
            
            BDD_nodes =  sorted(target_nodes)       # get all the ordered nodes that are indeed reached from the root node
            leaf_nodes = { (self.net_width * (self.net_depth-1)) + 1, (self.net_width * (self.net_depth-1)) + 2 }
            # BDD_levels = sorted(set([ (int((inner_node-1)/self.net_width+1) if inner_node else 0)  for inner_node in list(BDD_nodes)[:-2] ])) # get the ordered levels contained in BDD 
            BDD_levels = sorted(set([ (int((inner_node-1)/self.net_width+1) if inner_node else 0)  for inner_node in list(set(BDD_nodes) - leaf_nodes) ])) # get the ordered levels contained in BDD


            # >> filter out the unused nodes & levels 
            for idx in range(len(interpret_transition)):
                if not (idx in BDD_nodes):
                    interpret_left[idx] = 0
                    interpret_right[idx] = 0

            feature_selected = set()
            for idx in range(self.net_depth):   
                if idx in BDD_levels:

                    feature_id, _ = self.find_argmax_not_in_set(self.enc_dec[idx], feature_selected)
                    interpret_dec[idx][feature_id] = 1
                    feature_selected.add(feature_id)

                        

            interpret_transition = interpret_left + interpret_right
           

            '''verify the ordering property''' 
            # check if each col of enc_dec has at most one element 1
            self.is_ordered = torch.all(torch.sum(interpret_dec, dim=0) <= 1).item() 
            # print('is_ordered:', self.is_ordered)
            
            '''BDD Visualization''' # get the .dot file from the above reduced matrix

            if SAVE_DOT:
                with open(file_name, "w") as file:
                    file.write("digraph G {\n")

                    todepict_edges = torch.nonzero(interpret_transition[:-2,:], as_tuple = False).tolist() # depict all the edges except that the ones starting from leaves

                    for node_pair in todepict_edges:
                        if interpret_left[node_pair[0], node_pair[1]]:
                            file.write("\tn{} -> n{};\n".format(node_pair[0], node_pair[1]))

                        if interpret_right[node_pair[0], node_pair[1]]:
                            file.write("\tn{} -> n{}[style=\"dotted\"];\n".format(node_pair[0], node_pair[1]))

                    dec_map = interpret_dec.argmax(dim=1) 
                    # target_nodes = list(target_nodes).sort()
                    for idx, node in enumerate(BDD_nodes):
                        if node==(self.net_width * (self.net_depth-1)) + 1:
                            file.write("\tn{}[label=\"{}\", xlabel=\"n{}\", shape=square]\n".format(node, "⊤", idx))
                        elif node==(self.net_width * (self.net_depth-1)) + 2: 
                            file.write("\tn{}[label=\"{}\", xlabel=\"n{}\", shape=square]\n".format(node, "⊥", idx))
                        else:   # declaration of the decision nodes
                            if atoms_name is None:
                                file.write("\tn{}[label=\"x{}\", xlabel=\"n{}\"]\n".format(node, dec_map[int((node-1)/self.net_width+1) if node else 0], idx))  
                            else:
                                file.write("\tn{}[label=\"{}\", xlabel=\"n{}\"]\n".format(node, atoms_name[dec_map[int((node-1)/self.net_width+1) if node else 0]], idx)) 

 

                    ordering = [ (atoms_name[dec_map[level]] if atoms_name else f'x{dec_map[level]}') for level in BDD_levels]   
                    file.write( "\n\tlabel=\"The feature ordering: [%s] \n\t%s \"\n" % (', '.join(ordering), file_name))

                    file.write("}")
                    

        nodes_num = len(BDD_nodes)
        self.interpret_left = interpret_left
        self.interpret_right = interpret_right
        self.interpret_dec = interpret_dec

        return self.is_ordered, nodes_num



    def interpret_faithful_OBDD_encoding(self):
        '''
        Interpreting the (reduced) faithful OBDD encoding from the trained model (approximate faithful OBDD encoding)

        '''
        self.bulid_OBDD_encoding(test_flag=True)
        self.is_ordered = True
        leaf_nodes = self.leaf_nodes

        with torch.no_grad():
            '''Preprocessing for getting faithful OBDD encoding'''    # Binarization: reset the maximal element 1's in dim1, and 0's the others 

            # >> reset the maximal element 1 in dim1 (select the last one if it is not unique)
            # can ensure the validity of the connection
            interpret_left = torch.zeros_like(self.enc_left)
            interpret_left[torch.arange(len(interpret_left)), torch.argmax(self.enc_left, dim=1)] = 1   
            interpret_right = torch.zeros_like(self.enc_right)
            interpret_right[torch.arange(len(interpret_right)), torch.argmax(self.enc_right, dim=1)] = 1

            interpret_dec = torch.zeros_like(self.enc_dec)

            # >> filter the nodes & levels that are indeed reached from the root node
            interpret_transition = interpret_left + interpret_right
            candidate_edges = torch.nonzero(interpret_transition, as_tuple = False).tolist()  # the ordered candidate edges obtained from the interpreted transition matrix

            target_nodes = set()   
            target_nodes.add(0)  # set the root node (i.e., the 0-th node) as a target node
            for node_pair in candidate_edges:
                # add the end-point of the edge as a new target node, if its sourse can reached from the root node but is not a leaf node (so that remove its loop) 
                if (node_pair[0] in target_nodes) and (node_pair[0] < len(interpret_transition)-2): 
                    target_nodes.add(node_pair[1])  # add the target node
            
            BDD_nodes =  sorted(target_nodes)       # get all the ordered nodes that are indeed reached from the root node
            # leaf_nodes = [ (self.net_width * (self.net_depth-1)) + 1, (self.net_width * (self.net_depth-1)) + 2 ]
            # BDD_levels = sorted(set([ (int((inner_node-1)/self.net_width+1) if inner_node else 0)  for inner_node in list(BDD_nodes)[:-2] ])) # get the ordered levels contained in BDD 
            BDD_levels = sorted(set([ (int((inner_node-1)/self.net_width+1) if inner_node else 0)  for inner_node in list(set(BDD_nodes) - set(leaf_nodes)) ])) # get the ordered levels contained in BDD
            # print('BDD_nodes:', BDD_nodes)
            # print('BDD_levels:', BDD_levels)

            # ensure that each (selected) level is associated with a unique feature
            feature_selected = set()
            for idx in range(self.net_depth):  
                if idx in BDD_levels:
                    # feature_id = torch.argmax(self.enc_dec[idx], dim=1)
                    feature_id, _ = self.find_argmax_not_in_set(self.enc_dec[idx], feature_selected)
                    interpret_dec[idx][feature_id] = 1
                    feature_selected.add(feature_id)
                   
            # >> verify the ordering property,  sound but not complete(when not every level occur in the BDD) [Fixed. we have filtered out the unused nodes & levels]
            # check if each col of enc_dec has at most one element 1
            self.is_ordered = torch.all(torch.sum(interpret_dec, dim=0) <= 1).item() 
            # print('is_ordered:', self.is_ordered)


            '''get the reduced faithful OBDD encoding'''
            # >> filter out the unused nodes (and hence levels) 
            for idx in range(len(interpret_transition)):
                if not (idx in BDD_nodes):
                    interpret_left[idx] = 0
                    interpret_right[idx] = 0
                        

            self.BDD_nodes = BDD_nodes
            self.BDD_levels = BDD_levels
                    
            self.interpret_left = interpret_left
            self.interpret_right = interpret_right
            self.interpret_dec = interpret_dec
            

        return self.is_ordered, len(self.BDD_nodes)




    def decode(self, atoms_name=None):
        '''
        Decoding the reduced faithful OBDD encoding to get  get a BDD representation

        NOTE: should first call "interpret_faithful_OBDD_encoding" to get reduced faithful OBDD encoding 

        '''
        interpret_left = self.interpret_left
        interpret_right = self.interpret_right
        interpret_dec = self.interpret_dec
        BDD_nodes = self.BDD_nodes
        leaf_nodes = self.leaf_nodes

        assert BDD_nodes is not None

        BDD = None

        with torch.no_grad():

            lev_map = lambda node: int((node-1)/self.net_width+1)  if node else 0   
            dec_map = lambda level: interpret_dec.argmax(dim=1)[level]   


            left_edges = torch.nonzero(interpret_left[:-len(leaf_nodes),:], as_tuple = False).tolist() # collect all the left edges except that the ones starting from leaves
            right_edges = torch.nonzero(interpret_right[:-len(leaf_nodes),:], as_tuple = False).tolist() # collect all the right edges except that the ones starting from leaves
            left_edges = {item[0]: item[1] for item in left_edges}
            right_edges = {item[0]: item[1] for item in right_edges}
            nodes_tags = {}
            for node in BDD_nodes:
                if node==leaf_nodes[0]:
                    nodes_tags[node] = "⊤"
                elif node==leaf_nodes[1]:
                    nodes_tags[node] = "⊥"
                else:
                    nodes_tags[node] = dec_map(lev_map(node))

            feature_names = []
            if atoms_name is not None:
                feature_names = atoms_name
            else:
                feature_names = [f"f_{i}" for i in range(self.atom_size)]
            BDD = BinaryDecisionDiagram(feature_names)
            BDD.bulid(BDD_nodes, nodes_tags, left_edges, right_edges)

            # dot_description = BDD.get_dot_description()
            # print(dot_description)

        return BDD



    def OBDD_eval(self, data_x):
        '''
        evaluate the satisfaction relation between an example and the interpreted OBDD [this acts on the reduced faithful OBDD encoding!]
        
        ''' 
        device=data_x.device
        batch_size, atom_size = data_x.size()

        net_width = self.net_width
        net_depth = self.net_depth

        with torch.no_grad():
            data_x_selected = self.interpret_dec.unsqueeze(0).matmul(data_x.unsqueeze(2))  # (dim: batch_size * net_depth * 1). select and reorder the data_x

            filter_left_mid = data_x_selected[:,1:,:].repeat(1, 1, net_width).to(device)    # (dim: batch_size * net_depth-1 * net_width).  the probability to movring left branch of each interal node, under an example

            filter_left = torch.cat((
                    data_x_selected[:,0:1], \
                    filter_left_mid.view(batch_size, (net_depth-1) * net_width, 1), \
                    torch.ones(batch_size, 2, 1).to(device) \
                ), dim=1)      # (dim: batch_size * transfer_matrix_size * 1).
            filter_right = 1 - filter_left

            transfer_matrix = self.interpret_left.mul(filter_left) + self.interpret_right.mul(filter_right)  # (dim: batch_size * transfer_matrix_size * transfer_matrix_size)

            root_reach = transfer_matrix[:,0:1,:]   # the (1-step) reachability of the root node
            to_relation = torch.ones_like(root_reach).to(device)    # reset 1 the element larger than 1 
            for step in range(1, net_depth+1):      # the (i-step) reachability of the root node
                if step==1:
                    next_root_reach = root_reach
                else:
                    cur_root_reach = next_root_reach 
                    next_root_reach = cur_root_reach.matmul(transfer_matrix) 
                    next_root_reach = torch.where(next_root_reach > 1, to_relation, next_root_reach)


            result = next_root_reach[:,0,-2:]

        return result
        

    # def score(self, dataset, acc=None, f1=None, prec=None, recall=None, pos_label=1, average='binary', model_type='interpreted'):
    def score(self, dataset, model_type='interpreted', pos_label=1, average='binary', threshold=0.5):
        '''
        Compute the metrics of the train-original/test-original/interpreted model.

        :param - dataset:   the whole dataset for classification
        :param - model_type:        (1)'train-original': the original model in train stage (which uses gumbel_softmax) is evaluated; 
                                    (2)'test-original': the original model in test stage (which uses softmax) is evaluated;
                                    (3)'interpreted': the interpreted model (i.e., OBDD) is evaluated;
        :param - pos_label:  defined as in sklearn. The class to report if average='binary'.
        :param - average:  defined as in sklearn. If None, the scores for each class are returned; otherwise the specific average result for the classes is reported.
        :param - threshold:  the threshold for classification if model_type is 'train/test-original' 
        
        :return - acc, f1, pre, recall:  the corresponding metric result
        :return - is_ordered, nodes_num: when model_type is 'interpreted', these information about the interpreted OBDD is return 

        '''
        with torch.no_grad():
            if model_type=='train-original':
                self.bulid_OBDD_encoding(test_flag=False)
            elif model_type=='test-original':
                self.bulid_OBDD_encoding(test_flag=True)
            elif model_type=='interpreted':
                # is_ordered, nodes_num = self.interpret_BDD(SAVE_DOT=False)    
                is_ordered, nodes_num = self.interpret_faithful_OBDD_encoding()     
            else:
                print("No such model_type!")
                exit()

            dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
            pred = []
            for data_x, data_y in dataloader:
                if model_type=='train-original' or model_type=='test-original':
                    pred_batch, _ = self.forward(data_x, REFRESH_ENCODING=False) 
                    pred_batch = pred_batch.gt(threshold)   
                else:
                    pred_batch = self.OBDD_eval(data_x)   
                pred += pred_batch[:, 0].cpu().numpy().tolist()     # here consider binary classification only
                
            label = dataset.labels.cpu().numpy().tolist()

            acc = accuracy_score(label, pred)
            f1 = f1_score(label, pred, pos_label=pos_label, average=average)
            prec = precision_score(label, pred, pos_label=pos_label, average=average, zero_division=0)
            recall = recall_score(label, pred, pos_label=pos_label, average=average)   

            f1_macro = f1_score(label, pred, average="macro")

            if model_type=='interpreted':
                # print([int(elem) for elem in pred]) 
                return acc, f1_macro, f1, prec, recall, is_ordered, nodes_num
            else:
                return acc, f1_macro, f1, prec, recall



