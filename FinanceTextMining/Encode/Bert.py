import torch
from transformers import BertTokenizer, BertModel
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
word_embeddings_dim = 768  # 嵌入维度


#输入句子列表，输出特征向量列表
class Encode_txt_bert:
    # 加载BERT模型和tokenizer

    def __init__(self, sentence_list):
        # 得到特征矩阵
        self.encoding_list = np.array([self._encode(sentence) for sentence in sentence_list])

    def _encode(self, sentence):
        input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # 转化为id
        outputs = model(input_ids)
        embeddings = outputs[0][:, 0, :]  # 取出[CLS]对应的嵌入向量
        #print(type(embeddings))
        #print(embeddings.shape)
        return embeddings.detach().numpy()
