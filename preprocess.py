import os
import numpy
from sklearn.utils import shuffle
from tqdm import tqdm
import pickle

dataset = 'wiki'

data_path = os.path.join(os.getcwd(), 'graph', dataset+'.txt')

with open(data_path, 'r') as f:
    lines = f.readlines()
    comment_lines = lines[:4]
    full_data = lines[4:]

sample_ratio = 0.5
shuffle(full_data)
sample_data = full_data[:int(len(full_data) * sample_ratio)]

node_set = set()
for ele in sample_data:
    node1 = ele.strip().split('\t')[0]
    node2 = ele.strip().split('\t')[1]
    node_set.add(node1)
    node_set.add(node2)

if not os.path.exists(os.path.join(os.getcwd(), 'output', '{}_node2idx.pkl'.format(dataset))):
    all_nodes_bytes = [bytes(str(int(ele)), encoding='utf-8') for ele in list(node_set)]
    node2idx = {}
    for i, w in enumerate(all_nodes_bytes):
      node2idx[w] = i
    f = open(os.path.join(os.getcwd(), 'output', '{}_node2idx.pkl'.format(dataset)), 'wb')
    pickle.dump(node2idx, f)
    f.close()
else:
    f = open(os.path.join(os.getcwd(), 'output', '{}_node2idx.pkl'.format(dataset)), 'rb')
    node2idx = pickle.load(f)
    f.close()

n_nodes = len(node_set)
n_edges = len(sample_data)
with open(os.path.join(os.getcwd(), 'graph', '{}_small.txt'.format(dataset)), 'w') as f:
    f.writelines('% dataset {}, n_nodes: {}, n_edges: {}\n'.format(dataset, n_nodes, n_edges))
    f.writelines('% fromID\ttoID\tsign\n')
    for ele in tqdm(sample_data):
        f.writelines(ele)

