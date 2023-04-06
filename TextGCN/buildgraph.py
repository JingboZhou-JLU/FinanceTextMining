import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import dgl
import scipy.sparse as sp
import nltk
from nltk.corpus import wordnet as wn
from math import log
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import torch as th
import jieba
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
#nltk.data.path.append("C:\\Users\\lenovo\\AppData\\Roaming\\nltk_data")
#nltk.download('omw')
#nltk.download('wordnet')
from scipy.spatial.distance import cosine
###############改分词策略split,人家英文单词间有空格直接split分词就行，你中文整什么split

class Encode_and_BuildGraph:

    def __init__(self, sentence_list, word_embeddings_dim):
        self.word_embeddings_dim = word_embeddings_dim
        self.sentence_list = sentence_list
        self.vocab, self.word_freq, self.vocab_size = self.bulid_vocab()
        self.word_doc_list = self.get_word_list()
        self.word_doc_freq = self.get_word_freq()
        self.word_id_map = self.get_word_id_map()
        self.word_vector_map = self.word_definations()
        # 得到特征矩阵
        self.encoding_matrix = self.encode()
        self.row, self.col, self.weight = self.build_edges()
        self.graph,self.e_w,self.n_f = self.get_graph()

    def get_bert_embedding(self,text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings.mean(dim=1).squeeze().numpy()

    def bulid_vocab(self):
        # 统计所有样本中出现的单词，生成单词列表 vocab
        word_freq = {}
        word_set = set()
        for doc_words in self.sentence_list:
            words = jieba.lcut(doc_words)
            for word in words:
                word_set.add(word)
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        vocab = list(word_set)
        vocab_size = len(vocab)
        return vocab,word_freq,vocab_size

    def get_word_list(self):
        word_doc_list = {}
        # 将每个单词出现在哪些文本中进行统计，生成词汇表 word_doc_list  单词：[单词出现过的文档]
        for i in range(len(self.sentence_list)):
            doc_words = self.sentence_list[i]
            words = jieba.lcut(doc_words)
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)
        return word_doc_list

    def get_word_freq(self):
        # 统计每个单词在多少个文本中出现过，生成单词频率字典 word_doc_freq  单词：词频
        word_doc_freq = {}
        for word, doc_list in self.word_doc_list.items():
            word_doc_freq[word] = len(doc_list)
        return word_doc_freq

    def get_word_id_map(self):
        # 生成单词ID表 word_id_map
        word_id_map = {}
        for i in range(self.vocab_size):
            word_id_map[self.vocab[i]] = i
        return word_id_map

    def word_definations(self):
        definitions = []
        word_vectors = {}
        for word in self.vocab:
            #print(type(word.strip()))
            embed=self.get_bert_embedding(word.strip())
            print(embed)
            definitions.append(embed)

        for i in range(len(self.vocab)):
            word = self.vocab[i]
            vector = definitions[i]
            word_vectors[word]=vector
            #print(word)
            #print(vector)

        return word_vectors

    def encode(self):
        # 为每个文本生成其文本特征向量
        row_x = []
        col_x = []
        data_x = []

        for i in range(len(self.sentence_list)):
            doc_vec = np.array([0.0 for k in range(self.word_embeddings_dim)])  # 生成文本i的初始特征向量（全是0）
            doc_words = self.sentence_list[i]  # 获得文本i的文本内容
            words = jieba.lcut(doc_words)
            doc_len = len(words)
#            print('here')
#            print(words)
#            print(doc_len)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    #print(doc_vec)
                    #print(word_vector)#.detach().numpy()
                    #print('doc_vec'+str(type(doc_vec)), end='')
                    #print(doc_vec.shape)
                    #print('word_vector'+str(type(word_vector)),end='')
                    #print(word_vector.shape)
                    doc_vec = doc_vec + word_vector  # 文本的特征向量目前=文本中每个词的特征向量的叠加

            for j in range(self.word_embeddings_dim):
                row_x.append(i)
                col_x.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len  #文本特征向量最终取值 （文本每个词特征向量的叠加/文本总词汇数）
        for i in range(self.vocab_size):
            for j in range(self.word_embeddings_dim):
                row_x.append(int(i + len(self.sentence_list)))
                col_x.append(j)
                data_x.append(self.word_vector_map[self.vocab[i]][j])
        # 生成所有数据(句子[前]和单词[后])的完整特征向量矩阵
        # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
        # print(len(self.sentence_list))

        x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
            len(self.sentence_list)+self.vocab_size, self.word_embeddings_dim))
#        print('encode生成的文本特征向量为：', end='')
#        print(x.shape)
        return x

    def build_edges(self):
        # 使用单词共现（单词-单词边缘）在节点之间建立边缘，采用逐点互信息 (PMI)（一种流行的词关联度量）来计算两个词节点之间的权重
        # word co-occurence with context windows
        # 使用固定大小的滑动窗口来收集同现统计信息
        window_size = 20
        windows = []

        for doc_words in self.sentence_list:  # 取乱序文档内容列表中的每个文档内容
            words = jieba.lcut(doc_words)
            length = len(words)
            if length <= window_size:  # 如果文档内含词小于window_size，则将该文档词语列表作为一个window放入windows
                windows.append(words)
            else:
                # print(length, length - window_size + 1)
                for j in range(
                        length - window_size + 1):  # 如果文档内含词大于window_size，则对该文档迭代生成（length - window_size + 1）个window_size大小的window
                    window = words[j: j + window_size]
                    windows.append(window)
                    # print(window)

        # 字典word_window_freq存储每个单词在多少个窗口中出现过（同一单词在一个窗口中出现多次时不叠加）
        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                # 在遍历窗口的过程中，它跟踪每个单词是否在该window已经出现过，如果已经出现过则跳过这个单词
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        '''
        统计每个窗口内的单词对（co-occurrence pairs）在整个文本集合中的出现次数。
        具体来说，对于每个窗口，首先遍历窗口内的每个单词，对于出现在窗口内的每个单词，更新该单词出现在窗口中的频率。
        然后，遍历窗口中的每对单词，将单词对表示为逗号分隔的字符串，并将该单词对添加到单词对计数字典中。
        由于单词对的计算是对称的，因此还要交换单词的位置，重复以上步骤。
        '''
        word_pair_count = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = self.word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = self.word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        # 这三个用于生成边权值矩阵  eg： matrix(row[1],col[1])=weight[1]，矩阵行列标0-train_size代表文档，>train_size代表单词
        row = []
        col = []
        weight = []

        # pmi as weights
        # 采用逐点互信息 (PMI)（一种流行的词关联度量）来计算两个词节点之间的权重

        num_window = len(windows)

        # 计算每个共现单词对之间的PMI
        # word_pair_count中存储每个共现对，weight中存储每个共现对的PMI
        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[self.vocab[i]]
            word_freq_j = word_window_freq[self.vocab[j]]
            # 计算词i和词j之间的PMI
            pmi = log((1.0 * count / num_window) /
                      (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
            if pmi <= 0:
                continue
            row.append(len(self.sentence_list) + i)
            col.append(len(self.sentence_list) + j)
            weight.append(pmi)

        # 根据文档中的单词出现（文档-单词边缘）在节点之间建立边缘
        '''
        一个文档节点和一个词节点之间的边的权重是文档中词的词频-逆文档频率（TF-IDF）
        其中词频是词在文档中出现的次数，逆文档频率是包含该词的文档数量的对数倒数
        '''
        # doc word frequency
        doc_word_freq = {}  # 记录每个文档和文档中每个词节点的出现次数 eg:    key：文档1，单词2  value：3

        for doc_id in range(len(self.sentence_list)):  # 从头遍历sentence_list中所有文本
            doc_words = self.sentence_list[doc_id]
            words = jieba.lcut(doc_words)  # 取得该文本中所有单词
            for word in words:
                word_id = self.word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1

        for i in range(len(self.sentence_list)):  # 对于第i个文本
            doc_words = self.sentence_list[i]
            words = jieba.lcut(doc_words)
            doc_word_set = set()
            for word in words:  # 遍历文本中每个词
                if word in doc_word_set:
                    continue
                j = self.word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                row.append(i)
                col.append(len(self.sentence_list) + j)
                idf = log(1.0 * len(self.sentence_list) /  # 计算该文档-单词边权值
                          self.word_doc_freq[self.vocab[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)

        node_size = len(self.sentence_list) + self.vocab_size
        adj = sp.csr_matrix(
            (weight, (row, col)), shape=(node_size, node_size))
        return row,col,weight

    def get_graph(self):
        #edges = th.tensor(self.row), th.tensor(self.col)
        weights = th.tensor(self.weight)  # 每条边的权重
        #g = dgl.graph(edges)
        #g.edata['edge_weight'] = weights  # 将其命名为 'w'
        #print(weights)


        # 将编码矩阵转换为稀疏Tensor并设置为节点特征向量
        # if sp.issparse(self.encoding_matrix):
        #     encoding_matrix = th.sparse_coo_tensor(
        #         self.encoding_matrix.nonzero(),
        #         self.encoding_matrix.data,
        #         self.encoding_matrix.shape,
        #         dtype=th.float32
        #     )
        #     # 转置 self.word_vectors 组成的矩阵
        #     word_matrix = th.tensor([self.word_vector_map[word] for word in self.vocab])
        #     print('单词矩阵为',type(word_matrix))
        #     print('特征矩阵为', type(encoding_matrix))
        #     print('单词矩阵大小', word_matrix.shape)
        #     print('特征矩阵大小', encoding_matrix.shape)
        #     # 将其与 encoding_matrix 进行水平拼接
        #     encoding_matrix = th.cat((word_matrix, encoding_matrix), dim=0)
        #     print('特征矩阵大小', self.encoding_matrix.shape)
        # else:
        #     # 转置 self.word_vectors 组成的矩阵
        #     word_matrix = th.tensor([self.word_vector_map[word] for word in self.vocab])
        #     encoding_matrix = th.tensor(
        #         self.encoding_matrix, dtype=th.float32)
        #     # 将其与 encoding_matrix 进行水平拼接
        #     encoding_matrix = th.cat((word_matrix, encoding_matrix), dim=1)
        #     print('特征矩阵大小', self.encoding_matrix.shape)

        coo = self.encoding_matrix.tocoo()
        indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float)
        x = torch.sparse_coo_tensor(indices, values, size=coo.shape)
        g = dgl.DGLGraph()
        g.add_nodes(len(self.sentence_list)+self.vocab_size)
        g.add_edges(th.tensor(self.row),th.tensor(self.col))
        #g.ndata['node_feat'] = x
        return g,weights,x

