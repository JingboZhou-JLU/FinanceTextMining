import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import numpy as np

# 加载USE模型
module_path = "/root/FinanceTextProcess/FinanceTextProcess/Model"
model = tf.saved_model.load(module_path)


class Encode_txt_use:

    def __init__(self, sentence_list):
        # 得到特征矩阵
        self.encoding_list = np.array([self._encode(sentence) for sentence in sentence_list])

    def _encode(self, sentence):
        # 将句子转换为Tensor
        input_tensor = tf.constant(sentence)
        # 对句子进行编码
        output_tensor = model(input_tensor)
        # 获取编码结果
        encoding = output_tensor.numpy()
        # 输出编码结果
        return encoding
