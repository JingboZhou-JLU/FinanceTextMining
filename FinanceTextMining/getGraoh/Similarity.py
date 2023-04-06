import numpy as np
import dgl
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class GraphS:

    def __init__(self, features):
        self.features = features
        self.graph = self.get_graph()

    def get_graph(self):
        similarity_matrix = cosine_similarity(self.features.detach().numpy())
        # 将相似度矩阵转换为邻接矩阵
        threshold = 0.5  # 相似度阈值
        matrix = (similarity_matrix > threshold).astype(int)
        # 构建图
        g = dgl.from_scipy(csr_matrix(matrix))
        return g
