import dd.autoref as _bdd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import utils.util as util
# import util as util

dec_node = lambda dec_var, left_child, right_child: (left_child & dec_var) | (~ dec_var & right_child)

class DecisionVertex(object):
    """
        class of decision vertex in Binary Decision Diagram
        Every decision vertex has a decision feature (variable) to assign, and exact two children
    """
    def __init__(self, feature_id=None):
        self.feature_id = feature_id    # the index of the feature in the total features list of the dataset
        self.left_child = None          # record the index of vertex (it link via the left edge) in the self.vertices 
        self.right_child = None

        self.number = None      # not necessary, just for recording the corresponding position index in the faithful OBDD encoding


    def get_feature_id(self):
        return self.feature_id
    
    def get_left_child(self):
        return self.left_child
    
    def get_right_child(self):
        return self.right_child

    def set_left_child(self, left_child):
        self.left_child = left_child
    
    def set_right_child(self, right_child):
        self.right_child = right_child


    def get_number(self):
        return self.number

    def set_number(self, number):
        self.number = number




class LeafVertex(object):
    """
        class of leaf vertex in Binary Decision Diagram
        There are at most two leaf vertex, labelled by ⊤ and ⊥
    """
    def __init__(self, tag="⊤"):
        assert tag == "⊤" or tag == "⊥"
        self.tag = tag
    
        self.number = None      # not necessary, just for recording the corresponding position in the faithful OBDD encoding


    def get_tag(self):
        return self.tag

    def get_number(self):
        return self.number

    def set_number(self, number):
        self.number = number








class BinaryDecisionDiagram(object):
    """
        class of Binary Decision Diagram, which contains several decision vertices and at most two leaf vertices.

    """

    def __init__(self, feature_names=None):
        self.vertices = {}
        self.feature_names = [] if feature_names is None else feature_names     # the list of feature names of the dataset
        self.root = None
        self.vertices_cnt = 0
        self.leaf_cnt = 0

        self.feature_ordering = []      # set to None, if BDD does not satisfy the ordering property

        self.true_tag = "⊤"
        self.false_tag = "⊥"




    def get_vertices_cnt(self):
        return self.vertices_cnt

    def get_ith_vertex(self, ith):
        assert ith < self.get_vertices_cnt() and ith >= -self.get_vertices_cnt()
        return self.vertices[ith]

    def is_leaf(self, obj):
        return isinstance(obj, LeafVertex)


    def get_feature_ordering(self):

        if self.feature_ordering is not None:
            feature_names_ordering = [ self.feature_names[feature_id] for feature_id in self.feature_ordering]
        else:
            feature_names_ordering = None

        return self.feature_ordering, feature_names_ordering



    def is_ordered(self):
        return False if self.feature_ordering is None else True



    def bulid(self, BDD_nodes, nodes_tags, left_edges, right_edges):
        '''
        Bulid BDD from the given 4-tuple representation    

        :param - BDD_nodes:     the ordered list of nodes (number) interpreted from the structure of OBDDNet
        :param - nodes_tags:    a dict in which each element {node_i : tag_i} records the tag of a node in BDD_nodes, where tag_i is a ⊤, ⊥ or a feature_id
        :param - left_edges:    a dict records all the left edges {node_i : node_j}
        :param - right_edges:    a dict records all the right edges {node_i : node_j}

        '''   
        assert len(BDD_nodes)>1

        _BDD_nodes =  sorted(BDD_nodes)
        assert BDD_nodes==_BDD_nodes        # ??
        assert nodes_tags[_BDD_nodes[-1]]=="⊤" or nodes_tags[_BDD_nodes[-1]]=="⊥"  # the last node in BDD_nodes must be a leaf node

        feature_ordering = []

        BDD_nodes_set = set(_BDD_nodes)
        features_cnt = len(self.feature_names)
        for idx, node in enumerate(_BDD_nodes):
            vertex = None
            tag = nodes_tags[node]
            if tag=="⊤" or tag=="⊥":
                vertex = LeafVertex(tag)
                self.leaf_cnt += 1
            else:
                try:
                    tag = int(tag)
                    vertex = DecisionVertex(tag)
                    assert (node in left_edges.keys()) and (left_edges[node] in BDD_nodes_set)
                    assert (node in right_edges.keys()) and (right_edges[node] in BDD_nodes_set)

                    left_child_id = _BDD_nodes.index(left_edges[node])
                    right_child_id = _BDD_nodes.index(right_edges[node])
                    vertex.set_left_child(left_child_id)   
                    vertex.set_right_child(right_child_id)

                    assert tag<features_cnt

                    if (feature_ordering is not None) and (tag not in set(feature_ordering)):
                        feature_ordering.append(tag)
                    else:
                        if(tag!=feature_ordering[-1]):  # check if it satisfies the ordering property
                            feature_ordering = None
                except:
                    raise ValueError()

            
            vertex.set_number(node)     # record the position number of the node in the faithful OBDD encoding

            self.vertices[idx] = vertex
            self.vertices_cnt += 1

        self.root = self.vertices[0]
        self.feature_ordering = feature_ordering



    def get_dot_description(self):

        dot_description = "digraph G {\n"
        nodes_dot_info = ""
        edges_dot_info = ""

        internal_vertices_cnt = self.vertices_cnt - self.leaf_cnt
        for idx in range(self.vertices_cnt):
            vertex = self.vertices[idx]
            if idx < internal_vertices_cnt:
                feature = self.feature_names[vertex.feature_id]
                nodes_dot_info += f"\tn{vertex.get_number()}[label=\"{feature}\", xlabel=\"n{idx}\"]\n"


                left_child_id = vertex.get_left_child()
                right_child_id = vertex.get_right_child()
                left_child = self.get_ith_vertex(left_child_id)
                right_child = self.get_ith_vertex(right_child_id)
                edges_dot_info += f"\tn{vertex.get_number()} -> n{left_child.get_number()};\n"
                edges_dot_info += f"\tn{vertex.get_number()} -> n{right_child.get_number()}[style=\"dotted\"];\n"
            else:
                nodes_dot_info += f"\tn{vertex.get_number()}[label=\"{vertex.tag}\", xlabel=\"n{idx}\", shape=square]\n"

        feature_ordering, feature_names_ordering = self.get_feature_ordering()
        if feature_ordering is not None:
            # ordering_info = f"\n\tlabel=\"The feature ordering: [{','.join(feature_names_ordering)}]\"" 
            ordering_info = f"\n\tlabel=\"The feature ordering: {str(feature_ordering)} \n The corresponding feature name: [{','.join(feature_names_ordering)}]\"" 

        else:
            ordering_info = f"\n\tlabel=\"The BDD does not satisfy the ordering property!\""

        dot_description = "digraph G {\n" + nodes_dot_info + edges_dot_info + ordering_info + "\n}"

        return dot_description





    def construct_bdd(self):
        '''
        convert an object of BDD here to the one in dd package (so that we can use directly its manipulation)
        '''
        feature_ordering, _ = self.get_feature_ordering()
        feature_name_s = [f'f{i}' for i in feature_ordering]
        nodes_cnt = self.get_vertices_cnt()
        
        dec_vars = {}   # each element is "feature_id : f_i", where f_i is the dec_var object with expr_content "fi"

        bdd = _bdd.BDD()
        for idx, feature_name in enumerate(feature_name_s):
            bdd.declare(feature_name)                   # declare the vars
            dec_vars[feature_ordering[idx]] = bdd.add_expr(feature_name)  # create all the dec_var objects (auxiliary node)

        true = bdd.add_expr(r'True')
        false = bdd.add_expr(r'False')

        v = [None for _ in range(nodes_cnt)]  # in the bdd to be constructed, it maintains the index of all the nodes of given BDD here
        for ith in range(nodes_cnt-1, -1, -1):
            vertex = self.get_ith_vertex(ith)
            if self.is_leaf(vertex):
                if vertex.get_tag()==self.true_tag:
                    v[ith] = true 
                elif vertex.get_tag()==self.false_tag:
                    v[ith] = false 
                else:
                    raise ValueError()
            else:
                feature_id = vertex.get_feature_id()
                dec_var = dec_vars[feature_id]
                left_child_id = vertex.get_left_child()
                right_child_id = vertex.get_right_child()

                v[ith] = dec_node(dec_var, v[left_child_id], v[right_child_id])

        bdd.collect_garbage()
        bdd_root = v[0]
        # bdd.dump('rooted.png', roots=[bdd_root])

        # print(bdd.vars.keys())
        # print(feature_name_s)
        return bdd, bdd_root, feature_name_s




    def merge_BDD(self, bdd, BDD_s, apply_s=[]): 
        '''
        @param bdd: a initial empty object of  _bdd.BDD() to be modified
        @param BDD_s: a list of objects of  BinaryDecisionDiagram defined here
        @param apply_s: a list of operators ("OR"/"AND") to be performed on the BDD_s iteratively: (BDD_s[0] \OPER{apply_s[0]} BDD_s[1]) \OPER{apply_s[1]} ...

        @return bdd_root: the novel bdd_root of the bdd object
        '''
        assert isinstance(bdd, _bdd.BDD())
        assert len(BDD_s)==len(apply_s)+1
        assert isinstance(BDD_s[0], BinaryDecisionDiagram)
        assert len(BDD_s)==2        # for brevity, we consider only such basic case. Merging multi-bdd can be achieved via iteratively call this function

 
        feature_id_total= []
        for BDD in BDD_s:
            BDD_feature_ordering,_ = BDD.get_feature_ordering()
            feature_id_total += BDD_feature_ordering
        feature_id_total = list(set(feature_id_total))      # remove the duplicate features NOTE: the feature ordering does not matter here, as later _bdd will make adaptive adjustment on it

        dec_vars = {}   # a dict in which each element is "feature_id : f{feature_id}"
        for feature_id in feature_id_total:
            feature_name = f'f{feature_id}'
            bdd.declare(feature_name)                           # declare all the vars 
            dec_vars[feature_id] = bdd.add_expr(feature_name)  # create all the dec_var objects (auxiliary node) in bdd

        true = bdd.add_expr(r'True')
        false = bdd.add_expr(r'False')

        bdd_root_s = []
        for BDD in BDD_s:
            nodes_cnt = BDD.get_vertices_cnt()


            v = [None for _ in range(nodes_cnt)]  # in the (partial_)bdd to be constructed, it maintains the index of all the nodes of given BDD here
            for ith in range(nodes_cnt-1, -1, -1):
                vertex = BDD.get_ith_vertex(ith)
                if BDD.is_leaf(vertex):
                    if vertex.get_tag()==BDD.true_tag:
                        v[ith] = true 
                    elif vertex.get_tag()==BDD.false_tag:
                        v[ith] = false 
                    else:
                        raise ValueError()
                else:
                    feature_id = vertex.get_feature_id()
                    dec_var = dec_vars[feature_id]
                    left_child_id = vertex.get_left_child()
                    right_child_id = vertex.get_right_child()

                    v[ith] = dec_node(dec_var, v[left_child_id], v[right_child_id]) 

            # bdd.collect_garbage()
            _bdd_root = v[0]            
            bdd_root_s.append(_bdd_root)
            bdd.dump(f'rooted{_bdd_root}.png', roots=[_bdd_root])

        bdd_root = bdd_root_s[0]
        for idx in range(len(apply_s)):
            if apply_s[idx] == "AND":
                bdd_root = bdd_root & bdd_root_s[idx+1]
            elif apply_s[idx] == "OR":
                bdd_root = bdd_root | bdd_root_s[idx+1]
            else:
                raise ValueError()

        # bdd.collect_garbage()
        bdd.dump('rooted.png', roots=[bdd_root])
       
        return bdd_root



    def bdd_op_BDD(self, bdd, bdd_root, BDD, apply="AND"):
        '''
        @param bdd: a object of  _bdd.BDD() which represents a BDD got before
        @param bdd_root: the bdd_root of the bdd object
        @param BDD: a object of  BinaryDecisionDiagram defined here
        @param apply: an operators ("OR"/"AND") to be performed on the bdd and BDD: bdd \OPER{apply} BDD

        @return bdd_root: the novel bdd_root of the bdd object

        NOTE: bdd and BDD do not necessarily has the same (or even compatible) feature ordering, as the _bdd itself will make adaptive adjustment such as reordering ?
        '''
        # assert isinstance(bdd, _bdd.BDD())
        assert isinstance(BDD, BinaryDecisionDiagram)



        BDD_feature_ordering,_ = BDD.get_feature_ordering()
        dec_vars = {}   # a dict in which each element is "feature_id : f{feature_id}"
        for feature_id in BDD_feature_ordering:
            feature_name = f'f{feature_id}'
            bdd.declare(feature_name)                           # supplement the declaration of the (new) vars appear in BDD
            dec_vars[feature_id] = bdd.add_expr(feature_name)  # create the corresponding dec_var objects (auxiliary node) in bdd

        true = bdd.add_expr(r'True')
        false = bdd.add_expr(r'False')

        nodes_cnt = BDD.get_vertices_cnt()

        v = [None for _ in range(nodes_cnt)]  # in the (partial) bdd to be constructed, it maintains the index of all the nodes of given BDD here NOTE: but bdd may change them later 
        for ith in range(nodes_cnt-1, -1, -1):
            vertex = BDD.get_ith_vertex(ith)
            if BDD.is_leaf(vertex):
                if vertex.get_tag()==BDD.true_tag:
                    v[ith] = true 
                elif vertex.get_tag()==BDD.false_tag:
                    v[ith] = false 
                else:
                    raise ValueError()
            else:
                feature_id = vertex.get_feature_id()
                dec_var = dec_vars[feature_id]
                left_child_id = vertex.get_left_child()
                right_child_id = vertex.get_right_child()

                v[ith] = dec_node(dec_var, v[left_child_id], v[right_child_id])

        _bdd_root = v[0]    # ? the (partial_)bdd here may do not persist the same feature ordering as BDD_feature_ordering any more
        # bdd.dump(f'rooted{_bdd_root}.png', roots=[_bdd_root])

        if apply == "AND":
                bdd_root = bdd_root & _bdd_root
        elif apply == "OR":
            bdd_root = bdd_root | _bdd_root
        else:
            raise ValueError()

        # bdd.collect_garbage()
        bdd.dump('rooted.png', roots=[bdd_root])
       
        return bdd, bdd_root






    def _bdd_predict(self, dataset, bdd, bdd_root):

        data = dataset.get_data()

        feature_name_s = list(bdd.vars.keys())    # each element is "f{feature_id}" where feature_id in feature_ordering
        # print(feature_name_s)
        feature_ordering = [int(feature_name[1:]) for feature_name in feature_name_s]
        
        prediction = []
        for exam in data:
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

        bdd.collect_garbage()
        # print(bdd)

        # print(prediction)
        return prediction


    def bdd_predict(self, dataset):
        bdd, bdd_root, feature_name_s = self.construct_bdd()
        # prediction = self._bdd_predict(dataset, bdd, bdd_root, feature_name_s)
        prediction = self._bdd_predict(dataset, bdd, bdd_root)

        label = dataset.labels.cpu().numpy().tolist()
        acc = accuracy_score(label, prediction)
        f1 = f1_score(label, prediction, average=None)
        prec = precision_score(label, prediction, average=None, zero_division=0)
        rec = recall_score(label, prediction,  average=None)

        print(f"acc {acc} | f1 {f1} | prec {prec} | recall {rec}")
        return prediction



    def bdd_predict_score(self, dataset, bdd, bdd_root):
        
        prediction = self._bdd_predict(dataset, bdd, bdd_root)

        label = dataset.labels.cpu().numpy().tolist()
        acc = accuracy_score(label, prediction)
        f1 = f1_score(label, prediction, average=None)
        f1_macro = f1_score(label, prediction, average="macro")
        prec = precision_score(label, prediction, average=None, zero_division=0)
        rec = recall_score(label, prediction,  average=None)

        print(f"acc: {acc} | f1_macro: {f1_macro} | f1: {f1} | prec: {prec} | recall: {rec}")
        return prediction, acc, f1_macro, f1, prec, rec





    
if __name__ == '__main__':

    BDD_nodes = [0, 1, 3 ,5 , 6] 
    nodes_tags = {0:0, 1:1, 3:2, 5:"⊤", 6:"⊥"}
    left_edges = {0:1, 1:3, 3:5}
    right_edges = {0:3, 1:6, 3:6}
    feature_names = ["f_1", "f_2", "f_3", "f_4"]

    BDD = BinaryDecisionDiagram(feature_names)
    BDD.bulid(BDD_nodes, nodes_tags, left_edges, right_edges)

    dot_description = BDD.get_dot_description()
    print(dot_description)

    nodes_cnt = BDD.get_vertices_cnt()
    feature_ordering, feature_names_ordering = BDD.get_feature_ordering()
    print(nodes_cnt)
    print(feature_ordering)
    print(feature_names_ordering)

    
    bdd, bdd_root, _ = BDD.construct_bdd()
    bdd_actual_size = util.bdd_actual_size(bdd, bdd_root)
    BDD_size = BDD.get_vertices_cnt()
    print(f"BDD_size: {BDD_size} vs. bdd_actual_size: {bdd_actual_size}")
   