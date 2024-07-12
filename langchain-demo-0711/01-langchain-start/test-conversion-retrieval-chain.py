from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatSparkLLM
spark_chat_model=ChatSparkLLM()

from langchain_community.chat_models import ChatZhipuAI
zhipu_chat_model=ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)



#完成模型的选用 统一封装为chat_model
#chat_model=spark_chat_model
chat_model=zhipu_chat_model
chat_model_stream=zhipu_chat_model_stream


#2．为LLM提供额外的数据-context（从网页加载
'''
2-0．为加载网页数据进行环境准备
使用 WebBaseLoader 从网页（百度百科）抓取数据
本质:调用Python第三方 BeautifulSoup4，抓取页面，除去所有的html标签，保留网页文本
此方式,只能抓取|适用于静态网页
需要pip install beautifulsoup4
需要设置环境变量
'''
from langchain_community.document_loaders import WebBaseLoader
loader=WebBaseLoader(
    web_path="https://baike.baidu.com/item/%E5%8E%9F%E7%A5%9E/23583622",
)
#2-2. 将网页的数据加载到Document
docs=loader.load()
#print("docs:",docs)
#print("------------------------------")



#文档-词向量-向量数据库
#3. 将Documents索引到向量存储中
'''
    需要两个组件:生成词向量的嵌入模型、进行向量存储的向量数据库
    完成嵌入模型的环境准备：mae-base   https://huggingface.co/moka-ai/m3e-base
    0  安装所需模块
        pytorch -cpu  但我安的好像是gpu
        sentence-transformers:pip install sentence-transformers
    1  m3e-base
        https://aistudio.baidu.com/datasetdetail/234251
        https://huqgingface.co/moka-ai/m3e-base/tree/main
        下载，放置模型文件
    2  使用嵌入模型
        生成嵌入模型的实例对象
            指定 设备
            生成/实例化 嵌入模型的实例化对象
                指定模型的位置

'''
#from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
#from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#这样会降级
#安装 pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_DEVICE="cuda"  # 设置嵌入设备的类型为GPU
embeddings=HuggingFaceEmbeddings(
    model_name="langchain-demo-0711\models\m3e-base-huggingface",  # 指定使用的模型名称
    model_kwargs={"device":EMBEDDING_DEVICE},  # 传递模型参数，指定设备类型
    show_progress=True,  # 显示进度条
)
#print("embeddings:",embeddings)
#print("----------------------------------------------")


'''
    完成向量数据库的环境准备 FAISS
    0  安装所需模块 cpu /  gpu
        pip install faiss-cpu
        使用嵌入模型将文档生成词向量，存储到FAISS中
    1-导包
        导入向量存储的包
        导入生成词向里的包
    2-生成词向量
        1-分词/切分
        2-对一个个的词进行词向量的生成 存入FAISS中

'''
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 生成分词/切分器
text_splitter=RecursiveCharacterTextSplitter()
# 对load进来的文档进行分词/切分
documents=text_splitter.split_documents(documents=docs)

from langchain_community.vectorstores import FAISS
vector=FAISS.from_documents(documents=documents,embedding=embeddings)

print("vector:",vector)
print("-----------------------------------------------------")

'''
4.  生成链：检索链
    在向量存储FAISS中，存储并索引了数据（从网页中提取的数据）
    作用：
        接受 传入一个问题
        查找相关的向量数据库中的文档
        将这些文档和传入的原始问题 一起传递到LLM
        要求回答 原始问题
    
    4-0.  导包
    4-1.  生成一个来源于|基于向量存储的检索器

'''

# 创建一个向量检索器实例
retriever = vector.as_retriever()

'''
    4-2.  链：更新链
        接受：最近的输入input+会话历史chat_history
        LLM接受原始问题 请求 响应 提取
'''

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

#生成chatmodel会话的提示词
prompt=ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),#消息占位符  保存历史信息
    ("user","{input}"),
    #("user","根据以上对话历史，生成一个检索查询，以便查找与对话相关的信息")
    ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
])


#生成含有历史信息的检索链  模型，向量检索器，提示模板
retriever_chain=create_history_aware_retriever(chat_model,retriever,prompt)


#继续对话，记住检索到的文档等信息
prompt_2=ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])


from langchain.chains.combine_documents import create_stuff_documents_chain#将文档列表传递给模型
documents_chain= create_stuff_documents_chain(
    llm=chat_model,
    prompt=prompt_2,
)
from langchain.chains.retrieval import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever_chain,documents_chain)

from langchain_core.messages import HumanMessage,AIMessage
chat_history=[
    HumanMessage(content="原神是什么时候公测的？"),
    AIMessage(content="原神是2020年9月28日公测的。"),
]
chat_history.append(
    HumanMessage(content="原神游戏发生在哪？")
)
chat_history.append(
    AIMessage(content="原神游戏发生在一个被称作“提瓦特大陆”的幻想世界。")
)

response=retrieval_chain.invoke({
    "chat_history":chat_history,
    "input":"介绍一下这款游戏"
})
#print("response:",response)
#print("=============================================================")
#print("response['context']:",response["context"])
#print("=============================================================")
#print("response['chat_history']:",response["chat_history"])
#print("=============================================================")
#print("response['input']:",response["input"])
#print("=============================================================")
#print("response['answer']:",response["answer"])
#print("=============================================================")
#print("try chat....")


while True:
    human_message = input("请输入有关原神的问题（输入 'end' 结束）：")
    if human_message == "end":
        break
    # human_message 是那个问题
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": human_message
    })
    ai_message = response["answer"]
    print("回答：", ai_message)
    # 手工追加 聊天记录
    chat_history.append(HumanMessage(content=human_message))
    chat_history.append(AIMessage(content=ai_message))
print("END....")