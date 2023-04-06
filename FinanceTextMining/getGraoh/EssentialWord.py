import dgl
import jieba
import numpy as np
from scipy.sparse import csr_matrix
import torch

class GraphE:

    def __init__(self, sentence_list):
        self.sentence_list = sentence_list
        self.vocab, self.word_freq, self.vocab_size = self.bulid_vocab()
        self.word_id_map = self.get_word_id_map()
        self.word_doc_list = self.get_word_list()
        self.graph = self.get_graph()

    def get_word_id_map(self):
        # 生成单词ID表 word_id_map
        word_id_map = {}
        for i in range(self.vocab_size):
            word_id_map[self.vocab[i]] = i
        return word_id_map

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
        word_doc_list = [[0 for i in range(len(self.sentence_list))] for j in range(self.vocab_size)]
        # 将每个单词出现在哪些文本中进行统计，生成表 word_doc_list
        for i in range(len(self.sentence_list)):
            doc_words = self.sentence_list[i]
            words = jieba.lcut(doc_words)
            for word in words:
                word_doc_list[self.word_id_map[word]][i] = 1
        return word_doc_list

    def get_graph(self):
        # 构建词共现矩阵
        cooccur_matrix = np.zeros((len(self.sentence_list), len(self.sentence_list)), dtype=np.int32)
        for encoding in self.word_doc_list:
            for i in range(len(self.sentence_list)):
                for j in range(i + 1, len(self.sentence_list)):
                    word_i, word_j = encoding[i], encoding[j]
                    if word_i and word_j:
                        cooccur_matrix[i][j] += 1
                        cooccur_matrix[j][i] += 1

        # 将词共现矩阵转换为邻接矩阵，然后使用 DGL 构建图
        adj_matrix = csr_matrix(cooccur_matrix)
        g = dgl.from_scipy(adj_matrix)
        return g

