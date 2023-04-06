import streamlit as st
import dgl
from matplotlib import pyplot as plt
from Clustering.GMM import cluster_GMM
from Cov.simple_gcn import GCN
from Encode.Bert import Encode_txt_bert
from Encode.USE import Encode_txt_use
from Topic.TopicGet import GetTopic
from getGraoh.EssentialWord import GraphE
from getGraoh.Similarity import GraphS
import torch
from Cov.W_gcn import GCNConv
from TextGCN.buildgraph import Encode_and_BuildGraph
from Clustering.km import cluster_KM
import re
import jieba as jb
import streamlit as st
import Global_var
from ProcessData import pipelinetxt

st.set_option('deprecation.showPyplotGlobalUse', False)
# def input_process(user_input):
#     for i in range(4):
#         Global_var.sl.append(user_input[i].split(" "))
#     # print("here is input_process")
#     # print(Global_var.sl)
def delete_stopwords(strl): #输入为['','','']，输出为删除了停用词的['','','']
    back=[]
    for sentence in strl:
        sentence = re.sub(u"[0-9\-;\[.\]]+", u'', sentence)
        sentence_seged = jb.cut(sentence.strip())
        stopwords = [line.strip() for line in open(r"/root/FinanceTextProcess/FinanceTextProcess/Topic/StopWords.txt", 'r', encoding='utf-8').readlines()]
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords and word.__len__() > 1:
                if word != '\t':
                    outstr += word
        back.append(outstr)
    return back

def Clust(use_textgcn,Encode_way,getGraph_way,Clust_way,cluster_num,clas):
    if len(Global_var.sl_clean[clas])==0:
        return []
    # print("here is Clust")
    #print(Global_var.sl_clean[clas])
    if (use_textgcn != 'Y'):
        # 法一：Bert
        if (Encode_way == 'Bert'):
            features = Encode_txt_bert(Global_var.sl_clean[clas])
        else:
            # 法二：USE
            features = Encode_txt_use(Global_var.sl_clean[clas])
    #print(features.encoding_list.shape)
    if (use_textgcn != 'Y'):
        if (getGraph_way == 'Similarity'):
            # 法一：特征相似度
            if (Encode_way == 'Bert'):
                g = GraphS(torch.tensor(features.encoding_list.squeeze(axis=1))).graph
            else:
                g = GraphS(torch.tensor(features.encoding_list.squeeze(axis=1))).graph
        else:
            # 法二：词共现
            g = GraphE(Global_var.sl_clean[clas]).graph

    if (use_textgcn != 'Y'):
        # 法一：Bert
        if (Encode_way == 'Bert'):
            # if (features.encoding_list.shape[1] == 1):
            #     encoding_list = torch.tensor(features.encoding_list.squeeze(axis=1))
            # else:
            #     encoding_list = torch.tensor(features.encoding_list)
            g.ndata['h'] = torch.tensor(features.encoding_list)
            # print('Bert')
            # print(features.encoding_list.shape)
            model = GCN(len(features.encoding_list[0][0]), int(len(features.encoding_list[0][0]) / 2))
            embed = model(g)

        else:
            # 将 numpy 数组转换为 PyTorch 张量
            if(features.encoding_list.shape[1]==1):
                encoding_list = torch.tensor(features.encoding_list.squeeze(axis=1))
            else:
                encoding_list = torch.tensor(features.encoding_list)
#            print('here')
#            print(encoding_list.shape)
            # 将特征张量放到与 DGL 图相同的设备上
            encoding_list = encoding_list.to(g.device)
            g.ndata['h'] = encoding_list
            # print('USE')
            # print(features.encoding_list.shape)
            model = GCN(len(features.encoding_list[0][0]), int(len(features.encoding_list[0][0]) / 2))
            embed = model(g)

    if (use_textgcn == 'Y'):
        g = Encode_and_BuildGraph(Global_var.sl_clean[clas], 768)
        gcn = GCNConv(g.word_embeddings_dim, 100)
        embed = gcn(g.graph, g.n_f, g.e_w)
        # print("节点卷积后表征向量：", embed)

    if (Clust_way == 'KMean'):
        # 法一：k-means
        if (Encode_way == 'Bert'):
            return(cluster_KM(embed.squeeze(dim=1), cluster_num, len(Global_var.sl_clean[clas])).get_result())
        else:
            return(cluster_KM(embed, cluster_num, len(Global_var.sl_clean[clas])).get_result())
    else:
        # 法二：GMM
        if(embed.shape[0]<=1):
            #直接用kmeans
            if (Encode_way == 'Bert'):
                return (cluster_KM(embed.squeeze(dim=1), cluster_num, len(Global_var.sl_clean[clas])).get_result())
            else:
                return (cluster_KM(embed, cluster_num, len(Global_var.sl_clean[clas])).get_result())
        else:
            if (Encode_way == 'Bert'):
                return(cluster_GMM(embed.squeeze(dim=1), cluster_num, len(Global_var.sl_clean[clas])).get_result())
            else:
                return(cluster_GMM(embed,cluster_num, len(Global_var.sl_clean[clas])).get_result())

def page1():
    st.set_page_config(layout="wide")

    st.markdown("<h1 style='text-align: center;'>Explore Text Clustering With Different Combinations Of Methods</h1>", unsafe_allow_html=True)
    st.write("\n")
    st.write("\n")
#    st.markdown("<center>Enter the number of texts and categories for classification</center>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Upload text file and enter the number of categories for classification</h5>", unsafe_allow_html=True)
    st.write("\n")
#    st.write("Enter texts and the number of categories for classification")
    # 分成两列
#    [['管理和努力减轻我们的业务对环境的影响是我们业务的一个核心要素。','','',''],['','','',''],['','','',''],['','','','']]
#     file_input = st.text_input("Text is entered here", "sentence1 sentence2 sentence3 ...")
    uploaded_file = st.file_uploader("Pick up a txt file：")
    if uploaded_file is not None:
        # To read file as string:
        string_data = uploaded_file.read().decode("utf-8")
        user_input=pipelinetxt.begin(string_data) #得到[[],[],[],[]]
    #    print(user_input)
    Global_var.cluster_num = st.text_input("The number of categories is entered here", "")
    st.write("\n")
    col1,col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)

    with col6:
        if st.button("Go Clustering"):
            Global_var.sl_clean=[]
            Global_var.sl=[]
            for i in range(4):
                Global_var.sl.append(user_input[i])
                Global_var.sl_clean.append(delete_stopwords(user_input[i]))
            # print("sl:")
            # print(Global_var.sl)
            # print("sl_clean:")
            # print(Global_var.sl_clean)
            st.session_state["page"] = "page2"
            st.experimental_rerun()

    st.write("\n")

    # text_output.markdown(
    #     f'Use_TextGCN: {Global_var.use_textgcn}\nEncode_way: {Global_var.Encode_way}\nGetGraph_way: {Global_var.getGraph_way}\nClust_way: {Global_var.Clust_way}')

    st.write("\n")
    st.markdown("<h5 style='text-align: center;'>You can try:</h5>",
                unsafe_allow_html=True)
    st.write("\n")

    col1, col2, col3, col4= st.columns(4)

    # 在每列中放置元素
    with col1:
        # st.write("Do you want to use TextGCN?")
        # st.markdown(
        #     f"<h6 style='text-align: center;'>Now: {Global_var.use_textgcn}</h6>",
        #     unsafe_allow_html=True)
        text_output = st.empty()
        text_output.markdown(f"Do you want to use TextGCN? &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.use_textgcn}</b>", unsafe_allow_html=True)
        with st.container():
            if st.button("Yes"):
                Global_var.use_textgcn = 'Y'
                text_output.markdown(
                    f"Do you want to use TextGCN? &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.use_textgcn}</b>",
                    unsafe_allow_html=True)

            if st.button("No"):
                Global_var.use_textgcn = 'N'
                text_output.markdown(
                    f"Do you want to use TextGCN? &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.use_textgcn}</b>",
                    unsafe_allow_html=True)

    with col2:
        # st.write("Choose encoding way")
        # st.markdown(
        #     f"<h6 style='text-align: center;'>Now: {Global_var.Encode_way}</h5>",
        #     unsafe_allow_html=True)
        text_output = st.empty()
        text_output.markdown(f"Choose encoding way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.Encode_way}</b>", unsafe_allow_html=True)
        with st.container():
            if st.button("Bert"):
                Global_var.Encode_way = 'Bert'
                text_output.markdown(
                    f"Choose encoding way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.Encode_way}</b>",
                    unsafe_allow_html=True)

            if st.button("USE"):
                Global_var.Encode_way = 'USE'
                text_output.markdown(
                    f"Choose encoding way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.Encode_way}</b>",
                    unsafe_allow_html=True)

    with col3:
        # st.write("Choose graph constructing way")
        # st.markdown(
        #     f"<h6 style='text-align: center;'>Now: {Global_var.getGraph_way}</h5>",
        #     unsafe_allow_html=True)
        text_output = st.empty()
        text_output.markdown(f"Choose graph constructing way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.getGraph_way}</b>", unsafe_allow_html=True)
        with st.container():
            if st.button("Embedding Similarity"):
                Global_var.getGraph_way = 'Similarity'
                text_output.markdown(
                    f"Choose graph constructing way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.getGraph_way}</b>",
                    unsafe_allow_html=True)

            if st.button("Co-occurrence Word"):
                Global_var.getGraph_way = 'Word'
                text_output.markdown(
                    f"Choose graph constructing way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.getGraph_way}</b>",
                    unsafe_allow_html=True)

    with col4:
        # st.write("Choose graph clustering way")
        # st.markdown(
        #     f"<h6 style='text-align: center;'>Now: {Global_var.Clust_way}</h6>",
        #     unsafe_allow_html=True)
        text_output = st.empty()
        text_output.markdown(f"Choose graph clustering way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.Clust_way}</b>", unsafe_allow_html=True)
        with st.container():
            if st.button("KMeans"):
                Global_var.Clust_way = 'KMean'
                text_output.markdown(
                    f"Choose graph clustering way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.Clust_way}</b>",
                    unsafe_allow_html=True)

            if st.button("GMM"):
                Global_var.Clust_way = 'GMM'
                text_output.markdown(
                    f"Choose graph clustering way &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{Global_var.Clust_way}</b>",
                    unsafe_allow_html=True)


def get_wc(cls,num,result):
    w=[]  #w里每个元素为一句话
    for i in range(len(result)):
        if result[i]==num:
            w.append(Global_var.sl[cls][i])
    if(len(w)!=0):
        topic_str=GetTopic(w).topic_str
        st.write("\n")
        topic = 'classification ' + str(Global_var.class_num)
        Global_var.class_num+=1
        st.title(topic)
        st.markdown(f"<b>KeyWords:</b>&nbsp;&nbsp;&nbsp;<span style='font-family:SimSun'>{topic_str}</span>", unsafe_allow_html=True)
        st.write("\n")
        index=1
        for i in w:
            st.markdown(f"{index}.<span style='font-family:SimSun'>{i}</span>", unsafe_allow_html=True)
            index+=1

def page2():
    #print("tap once and in")
    if Global_var.cluster_num == '':
        Global_var.cluster_num=3
    else:
        Global_var.cluster_num = int(Global_var.cluster_num)
    result=[]
    for i in range(4):
        result.append(Clust(Global_var.use_textgcn, Global_var.Encode_way, Global_var.getGraph_way, Global_var.Clust_way, Global_var.cluster_num,i))
    st.markdown("<h1 style='text-align: center;'>Clustering Result as Follow</h1>",
                unsafe_allow_html=True)
    st.write("\n")
    dic=['Environmental','Social','Governance','Other']
    for j in range(4):
        if len(result[j])!=0:
            st.markdown(f"<h3 style='text-align: center;'>{dic[j]}</h3>", unsafe_allow_html=True)
            for i in range(Global_var.cluster_num):
                get_wc(j,i,result[j])
            Global_var.class_num = 1
    st.write("\n")
    st.write("\n")
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)

    with col6:
        if st.button("Return"):
            st.session_state["page"] = "page1"
            st.experimental_rerun()

def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "page1"

    pages = {
        "page1": page1,
        "page2": page2,
    }

    current_page = pages[st.session_state.get("page", "page1")]
    current_page()

if __name__ == "__main__":
    main()
