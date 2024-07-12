from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatSparkLLM
spark_chat_model=ChatSparkLLM()

from langchain_community.chat_models import ChatZhipuAI
zhipu_chat_model=ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
    #streaming=True,
    #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)


#完成模型的选用 统一封装为chat_model
#chat_model=spark_chat_model
chat_model=zhipu_chat_model


#0. 构建没有答案的message
message_str="数据空间研究院是谁出资建立的？"
response=chat_model.invoke(input=message_str)
print("chat_model.content:",response.content)
print("----------------------------------------------")

#1. 为llm提供额外的数据-content
from langchain.prompts.chat import(
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
#1-1 生成含有上下文|领域数据提示（占位符）的chat_model提示词模板
chat_model_prompt_message_template=ChatPromptTemplate.from_template(
    """
    仅根据提供的上下文回答以下问题
    <上下文>
    {context}
    </上下文>

    问题：{input}
    """
)
#1-2. 生成一个链，数据源是文档、提示词模板中的文本
from langchain.chains.combine_documents import create_stuff_documents_chain
documents_chain= create_stuff_documents_chain(
    llm=chat_model,
    prompt=chat_model_prompt_message_template,
)

#1-3. 传入文档信息：手动传入，从指定位置（网页、本地文件、数据库）加载
#手动传入
from langchain_core.documents import Document
documents_chain.invoke(
    input={
        "input":message_str,
        "context":[
            Document(
               page_content="合肥综合性国家科学中心数据空间研究院是由安徽省人民政府发起成立的事业单位，"
               "是新一代信息技术数据空间领域的大型新型研发机构，致力于引领网络空间安全和数据要素创新技术前沿和创新方向，"
               "凝聚一批海内外领军科学家团队，汇聚相关行业大数据，开展数据空间基础理论、体系架构、关键技术研究以及相关能力建设，"
               "打造大数据发展新高地，推进“数字江淮”建设，为数字中国建设贡献“安徽智慧”“合肥智慧”。"
            ),
        ]
    }
)
print("response:",response)
print("response.content:",response.content)
print("--------------------------------------------------------")

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
    web_path="https://baike.baidu.com/item/%E5%90%88%E8%82%A5%E7%BB%BC%E5%90%88%E6%80%A7%E5%9B%BD%E5%AE%B6%E7%A7%91%E5%AD%A6%E4%B8%AD%E5%BF%83%E6%95%B0%E6%8D%AE%E7%A9%BA%E9%97%B4%E7%A0%94%E7%A9%B6%E9%99%A2/62996254",
)
#2-2. 将网页的数据加载到Document
docs=loader.load()
print(docs)

#文档-词向量-向量数据库
#3. 将Documents索引到向量存储中
'''
    需要两个组件:生成词向里的嵌入模型、进行向里存储的向里数据库
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
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#这样会降级
#安装 pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_DEVICE="cuda"  # 设置嵌入设备的类型为GPU
embeddings=HuggingFaceEmbeddings(
    model_name="langchain-demo-0711\models\m3e-base-huggingface",  # 指定使用的模型名称
    model_kwargs={"device":EMBEDDING_DEVICE},  # 传递模型参数，指定设备类型
)
print("embeddings:",embeddings)
print("----------------------------------------------")
'''
    完成向量数据库的环境准备 FAISS
    0  安装所需模块 cpu /  gpu
        pip install faiss-cpu
        使用嵌入模型将文档生成词向里，存储到FAISS中
    1-导包
        导入向量存储的包
        导入生成词向里的包
    2-生成词向量
        1-分词/切分
        2-对一个个的词进行词向量的生成 存入faiss

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
    4-2.  生成一个检索链，包含检索器+原始问题
    4-3.  运行检索链，给出原始问题中占位符的替换的真实的问题字符串
          发送请求 获取响应
    4-4.  响应的结果
'''


from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()
#retrieval_chain封装两个参数：检索器retriever、domain_chain原始问题（带有占位符的提示模板）
retrieval_chain = create_retrieval_chain(retriever, documents_chain)
response = retrieval_chain.invoke({"input": message_str})
print(response)
print(response["answer"])

print("end...")

'''
创建 检索链，检索知识，准备：嵌入模型、向量存储
？？？ 仅支持单轮提问/回答， 交互
？？？ 多轮会话？？？
'''