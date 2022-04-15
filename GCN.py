import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import networkx as nx
import Load_Karate_Club
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from Load_Karate_Club import graph, X_train, Y_train
from sklearn.metrics import classification_report
from Load_Karate_Club import X_test, Y_test

# H^(l+1) = σ(D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l) * W^(l))
# A_hat = A + I 其中I为单位矩阵
# D_hat为A_hat的degree matrix
# H为每一层的特征，对于输入层，H就是X
# σ是非线性激活函数

'''建立GCN模型'''
class GCN_layer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(GCN_layer, self).__init__()
        # 随机初始化权重, torch.rand()生成a*b的均匀tensor
        self.W = Parameter(torch.rand(input_shape, output_shape))
        # 随机初始化bias
        self.bias = Parameter(torch.rand(output_shape))

    def forward(self, Adj_matrix, input_feature):
        # 将Adj_matrix从numpy变成tensor
        A = torch.from_numpy(Adj_matrix).type(torch.LongTensor)
        # 判断矩阵A是否为正方形矩阵
        assert A.shape[0]==A.shape[1]
        I = torch.eye(A.shape[0])
        A_hat = A + I
        # 对A_hat在dim=0方向上进行求和
        D = torch.sum(A_hat, 0)
        D = torch.diag(D)  # 以D建对角矩阵
        print(D.shape)
        D_inv = torch.sqrt(torch.inverse(D))  # 求逆矩阵再求根号, D^(-1/2)
        A_hat = torch.mm(torch.mm(D_inv, A_hat), D_inv)  # D^(-1/2)*A_hat*D^(-1/2)

        aggregate = torch.mm(A_hat, input_feature)  # D^(-1/2)*A_hat*D^(-1/2)*H^(l)
        propagate = torch.mm(aggregate, self.W) + self.bias
        # 返回H^(l+1)
        return propagate

# GCN = GCN_layer(36,4)
# Adj_matrix = np.arange(34*34).reshape(34,34)
# input_feature = torch.rand(34,36)

# #propagate=GCN.forward(Adj_matrix,input_feature)
# propagate=GCN(Adj_matrix,input_feature)
# print(propagate.shape)
#打印模型的参数
# for param in GCN.parameters():
#     print(param.data, param.size())

class GCN(nn.Module):
    def __init__(self, input_shape, output_shape, n_classes, activation='Relu'):
        super(GCN, self).__init__()
        # 第一层
        self.layer1 = GCN_layer(input_shape, output_shape)

        # 第二层
        self.layer2 = GCN_layer(output_shape, n_classes)

        # 激活函数
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax()
        elif activation == 'Relu':
            self.activation = nn.ReLU()

        # 在dim=1的方向和为1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Adj_matrix, input_feature):
        x = self.layer1(Adj_matrix, input_feature)
        x = self.activation(x)
        x = self.layer2(Adj_matrix, x)
        x = self.softmax(x)
        return x
# 输出的维度为 节点个数*n_classes
# y = model(Adj_matrix, input_feature)

def create_features(graph):
    A = np.array(nx.to_numpy_matrix(graph))  # 邻接矩阵
    X_1 = torch.eye(A.shape[0])
    X_2 = torch.zeros(A.shape[0], 2)  # 生成A.shape[0]*2的零矩阵
    node_distance_instructor = nx.shortest_path_length(graph, target=33)
    node_distance_administrator = nx.shortest_path_length(graph, target=0)

    # 为每个节点生成到Admin和Instr的最短距离
    # 在X_2中，[0]列为到admin的最短距离，[1]列为到Instr的最短距离
    for node in graph.nodes():
        X_2[node][0] = node_distance_administrator[node]
        X_2[node][1] = node_distance_instructor[node]
    return torch.cat((X_1, X_2),dim=1)

print(create_features(Load_Karate_Club.graph))
print(create_features(Load_Karate_Club.graph).shape)  # 34*36矩阵

# 要求输入：input_shape=特征的列数(也就是36)  output_shape可选
model = GCN(input_shape=create_features(graph).shape[1], output_shape=4, n_classes=2, activation='Tanh')
A = np.array(nx.to_numpy_matrix(graph))  # adjacency matrix 邻接矩阵
# print(A.shape)

'''开始训练'''
class Trainer():
    def __init__(self, model, optimizer, loss_function, epochs):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs

    def train(self, X_train, Y_train):
        epoch_loss_array = []
        tot_loss_array = []
        y_train = torch.from_numpy(Y_train.astype(int)).type(torch.LongTensor)
        tot_loss = 0.0

        all_preds = []

        for t in range(self.epochs):
            epoch_loss = 0.0

            # model.train()

            self.optimizer.zero_grad()
            # 在model中输入邻接矩阵和特征矩阵
            y_pred = self.model(A, create_features(graph))  # 每一个节点的预测值（二分类）
            # print(y_pred)
            # print(y_pred.shape)
            all_preds.append(y_pred)
            # print(all_preds)
            # print(y_pred[X_train])
            #  print(y_train)
            loss = self.loss_function(y_pred[X_train], y_train)  # 得到损失函数
            # print(loss)
            # print(loss.shape)

            epoch_loss += loss
            tot_loss += loss

            loss.backward()
            self.optimizer.step()
            print(str(t), 'epoch_loss:' + str(epoch_loss), 'total loss:' + str(tot_loss))

            # 将epoch_loss和tot_loss和计算图脱离
            # 返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算，即requires_grad=False
            # 修改其中一个tensor的值，另一个也会改变，因为是共享同一块内存
            # 但如果对其中一个tensor执行某些内置操作，则会报错，例如resize_、resize_as_、set_、transpose_
            temp = epoch_loss.detach().numpy()
            temp1 = tot_loss.detach().numpy()

            epoch_loss_array.append(temp)
            tot_loss_array.append(temp1)
        # print(epoch_loss_array,type(epoch_loss_array))
        x = np.arange(len(epoch_loss_array))
        # print(x)
        # print(epoch_loss_array, type(epoch_loss_array))

        plt.subplot(1, 2, 1)
        plt.plot(x, epoch_loss_array, 'b')
        plt.xlabel('epochs')
        plt.ylabel('epoch_loss_array')

        x2 = np.arange(len(tot_loss_array))
        plt.subplot(1, 2, 2)
        plt.plot(x2, tot_loss_array, 'r')
        plt.xlabel('epochs')
        plt.ylabel('tot_loss_array')
        plt.show()
        self.all_preds = all_preds

    def test(self, X_test, Y_test):
        self.model.eval()
        y_test = torch.from_numpy(Y_test.astype(int)).type(torch.LongTensor)
        y_pred = self.all_preds[-1]  # 最后一次epoch预测的结果

        loss_test = self.loss_function(y_pred[X_test], y_test)
        print('validation loss is equal to: ' + str(loss_test))

    # 可视化分类
    def visualize_classification(self, graph, Y_test, classification):
        last_epoch = self.all_preds[self.epochs - 1].detach().numpy()  # get outputs of last epoch
        # predicted_class为返回矩阵中每行数值大的index
        predicted_class = np.argmax(last_epoch, axis=-1)  # take the unit with the higher probability
        # 如果分类为0，则显示为红色
        color = np.where(predicted_class == 0, 'c', 'r')
        # Position nodes using Kamada-Kawai path-length cost-function 使用路径长度cost function定位节点位置
        pos = nx.kamada_kawai_layout(graph)
        nx.draw_networkx(graph, pos, node_color=color, with_labels=True, node_size=300)
        plt.show()
        if classification == True:
            print(classification_report(predicted_class[1:-1], Y_test))
            # print("true")


'''开始训练'''
trainer = Trainer(
    model,
    optimizer=optim.Adam(model.parameters(), lr=0.01),
    loss_function=F.cross_entropy,
    epochs=250
)
trainer.train(X_train, Y_train)

'''开始测试'''
trainer.test(X_test, Y_test)

trainer.visualize_classification(graph, Y_test, classification=True)