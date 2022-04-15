import torch
import torch.nn as nn
import numpy as np
import networkx as nx  # 复杂网络分析库
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

# 生成一个属性图
def create_Graphs_with_attributes(edgelist_filepath, attributes_filepath):
    graph = nx.read_edgelist(edgelist_filepath, nodetype=int)
    # 读取节点的属性，并设置属性表的索引列，名为node
    attributes = pd.read_csv(attributes_filepath)
    # 获取属性值（字典）
    attributes_values = {a:{'role':b[1], 'community':b[2]} for a, b in enumerate(attributes.values)}
    # 为图设置节点的属性
    nx.set_node_attributes(graph, attributes_values)
    return graph

graph = create_Graphs_with_attributes('karate.edgelist.txt', 'karate.attributes.csv')
nx.draw(graph, with_labels=True, node_color=(0.5294, 0.804, 0.9215), node_size=500)
plt.show()

# 生成节点邻接矩阵
A = np.array(nx.to_numpy_matrix(graph))
print(A)

# 生成训练集、测试集
def creat_train_test(graph):
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for node, data in graph.nodes(data=True):
        if data['role'] in ['Administrator', 'Instructor']:
            X_train.append(node)
            Y_train.append(data['role']=='Administrator')  # 是Admin返回True，不是返回False
        else:
            X_test.append(node)
            Y_test.append(data['community']=='Administrator')
    return np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test)
X_train, Y_train, X_test, Y_test = creat_train_test(graph)
# 将两个人提取到train中，剩下的member都在test中，与Admin联系的返回True，否则为False