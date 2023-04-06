from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import requests
import hashlib
import json
import random

class TextAnalysisMech(object):
    def __init__(self,list):
        self.Env = []
        self.Soc = []
        self.Gov = []
        self.Non = []
        self.finlist = list
        self.fineng = []
        # finbert
        self.finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
        # 翻译服务
        self.appid = '20230405001629540'
        self.secretKey = 'Q_NmcI36fKvhaLfU6Eco'
        for  s in self.finlist:
            self.fineng.append(self.translate(s))


    def translate(self,q, from_lang='zh', to_lang='en'):
        url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5((self.appid + q + salt + self.secretKey).encode('utf-8')).hexdigest()
        params = {
            'q': q,
            'from': from_lang,
            'to': to_lang,
            'appid': self.appid,
            'salt': salt,
            'sign': sign
        }
        response = requests.get(url, params=params)
        result = json.loads(response.text)
        return result['trans_result'][0]['dst']

    def classification(self):
        nlp = pipeline("text-classification", model=self.finbert, tokenizer=self.tokenizer)
        results = nlp(self.fineng)
        for ind in range(len(results)):
            if results[ind]['label'] == 'Environmental':
                self.Env.append(self.finlist[ind])
            elif results[ind]['label'] == 'Social':
                self.Soc.append(self.finlist[ind])
            elif results[ind]['label'] == 'Governance':
                self.Gov.append(self.finlist[ind])
            else:
                self.Non.append(self.finlist[ind])
        return [self.Env,self.Soc,self.Gov,self.Non]


import re

import jieba


class CutTextMech(object):
    def __init__(self,s):
        # 处理的文本文件
        self.text = s
        # 分句
        self.sentence = []
        # 分词
        self.words = []



    def RetSen(self):
        # 分句
        syb = '(。|！|\!|\.|？|\?)'
        self.sentence = re.split('(。|！|\!|？|\?|\n| )',self.text)
        self.sentence = [i for i in self.sentence if len(i) > 1]
        return self.sentence

    def RetWrd(self):
        # 分词
        with open('stop_word.txt', 'r', encoding='utf-8') as f:
            stopword_list = [word.strip('\n') for word in f.readlines()]
        for s in self.sentence:
            s = ''.join(filter(str.isalpha, s))
            wl = jieba.lcut(s)
            wl = [i for i in wl if i not in stopword_list]
            wl = [i for i in wl if len(i) >0]
            self.words.append(wl)
        return self.words





def begin(string_):
    # 生成文本处理器
    ctx = CutTextMech(string_)
    # 计算分词结果
    ctx.RetSen()
    # 生成分类处理器
    return TextAnalysisMech(ctx.sentence).classification()