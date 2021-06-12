import utils as u
import os
import torch
#erase
import time
import tarfile
import itertools
import numpy as np

class Elliptic_Temporal_Dataset:
    def __init__(self,args):
        args.elliptic_args = u.Namespace(args.elliptic_args)

        tar_file = os.path.join(args.elliptic_args.folder, args.elliptic_args.tar_file)
        tar_archive = tarfile.open(tar_file, 'r:gz')

        self.nodes, self.nodes_feats = self.load_node_feats(args.elliptic_args, tar_archive)

        self.nodes_labels_times = self.load_node_labels(args.elliptic_args, tar_archive)

        self.edges = self.load_transactions(args.elliptic_args, tar_archive)



    def load_node_feats(self, elliptic_args, tar_archive):
        data = u.load_data_from_tar(elliptic_args.feats_file, tar_archive, starting_line=0, replace_unknow=True)
        nodes = data
        print("Nodes", data.shape)

        nodes_feats = nodes[:,1:]
        print("Nodes Feats", nodes_feats.shape)

        self.num_nodes = len(nodes[:,0])
        print("Num Nodes", self.num_nodes)
        self.feats_per_node = data.size(1) - 1
        print("Feats Per node", self.feats_per_node)

        return nodes, nodes_feats.float()


    def load_node_labels(self, elliptic_args, tar_archive):
        labels = u.load_data_from_tar(elliptic_args.classes_file, tar_archive, replace_unknow=True).long()
        times = u.load_data_from_tar(elliptic_args.times_file, tar_archive, replace_unknow=True).long()
        lcols = u.Namespace({'nid': 0,
                             'label': 1})
        tcols = u.Namespace({'nid':0, 'time':1})


        nodes_labels_times =[]
        for i in range(len(labels)):
            label = labels[i,[lcols.label]].long()
            if label>=0:
                nid=labels[i,[lcols.nid]].long()
                time=times[i,[tcols.time]].long()
                nodes_labels_times.append([nid , label, time])
        nodes_labels_times = torch.tensor(nodes_labels_times)
        new_nodes = nodes_labels_times[:,0]
        _, new_nodes = new_nodes.unique(return_inverse=True)
        nodes_labels_times[:, 0] = new_nodes
        print("Nodes_Labels_times", nodes_labels_times)
        print("Nodes Label Times shape", nodes_labels_times.shape)

        return nodes_labels_times


    def load_transactions(self, elliptic_args, tar_archive):
        tcols = u.Namespace({'source': 0,
                             'target': 1,
                             'time': 2})

        data = u.load_data_from_tar(elliptic_args.edges_file, tar_archive, type_fn=float, tensor_const=torch.LongTensor)
        # print("EDGES", data)
        new_edges = data[:, [tcols.source, tcols.target]]
        _, new_edges = new_edges.unique(return_inverse=True)
        data[:, [tcols.source, tcols.target]] = new_edges
        print("contigous edges", data.shape)

        data = torch.cat([data,data[:,[1,0,2]]])
        print("Final Data",data.shape)

        self.max_time = data[:,tcols.time].max()
        # print("Max Time",self.max_time)
        self.min_time = data[:,tcols.time].min()
        # print("Min Time", self.min_time)

        return {'idx': data, 'vals': torch.ones(data.size(0))}




