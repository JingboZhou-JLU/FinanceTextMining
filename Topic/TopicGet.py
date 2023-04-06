import re
import jieba as jb
import gensim
from gensim import models
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import codecs

class GetTopic:
    def __init__(self, sentence_list):
        self.topic_str=self.list_express(sentence_list)

    def stopwordslist(self, filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords

    def seg_sentence(self, sentence): #输入一句话，返回分词后以空格分割的str
        sentence = re.sub(u"[0-9\-;\[.\]]+", u'', sentence)
        sentence_seged = jb.cut(sentence.strip())
        stopwords = self.stopwordslist(
            r"/root/FinanceTextProcess/FinanceTextProcess/Topic/StopWords.txt")  # 加载停用词的路径
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords and word.__len__() > 1:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr


    def list_express(self, sentence_list):
        train = []
        for sentence in sentence_list:
            line_seg = self.seg_sentence(sentence)  # 这里的返回值是字符串
            line = line_seg.split()
            train.append([w for w in line])

        dictionary = corpora.Dictionary(train)
        corpus = [dictionary.doc2bow(text) for text in train]

        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, passes=60)
        # num_topics：主题数目
        # passes：训练伦次
        # num_words：每个主题下输出的term的数目
        topic_str=''
        for topic in lda.print_topics(num_words=3):
            listOfTerms = topic[1].split('+')
            for term in listOfTerms:
                listItems = term.split('*')  #listItems[0]为权值
                topic_str+=listItems[1]
        return topic_str

#topic=GetTopic(['张三去医院看病', '李四去诊所看病','老刘生病了得去看看是什么情况'])
