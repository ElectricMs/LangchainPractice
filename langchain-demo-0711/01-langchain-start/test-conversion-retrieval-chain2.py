from dotenv import load_dotenv
load_dotenv()

# 生成/构建 语言模型- Chat Model
'''
    导包 + 实例化
    可能还需要 安装 python 第三方模块
'''
# sparkllm model
from langchain_community.chat_models import ChatSparkLLM

spark_chat_model = ChatSparkLLM()
# zhipuai model
from langchain_community.chat_models import ChatZhipuAI

zhipuai_chat_model = ChatZhipuAI(model="glm-4")
# 完成模型的选用，统一封装为 chat_model
# chat_model = spark_chat_model
chat_model = zhipuai_chat_model

# 0. 构建 没有答案 的消息字符串
message_str = "数据空间研究院是谁出资建立的？"
response = chat_model.invoke(input=message_str)
print("chat model: ", response)
print("chat model content: ", response.content)
print("===============================================")

# 1. 为 LLM 提供额外的数据-context（放在提示词模板中）
# 1-0. 导包
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 1-1 生成含有上下文|领域数据 提示（占位符）的 chat_model 提示词模板
chat_model_prompt_message_template = ChatPromptTemplate.from_template(
    """仅根据提供的上下文回答以下问题：
    <上下文>
    {context}
    </上下文>

    问题：{input}
    """
)
# 1-2 生成一个链（DocumentChain），数据源是 文档（本地各种格式的文档：txt\doc\pdf...）、提示词模板中的文本...
from langchain.chains.combine_documents import create_stuff_documents_chain

documents_chain = create_stuff_documents_chain(llm=chat_model, prompt=chat_model_prompt_message_template)
# 1-3 传入文档消息：手动传入、从指定的位置（网页、本地文件=txt\doc\pdf...)加载
# 手动传入，发送、获取响应并输出
from langchain_core.documents import Document

response = documents_chain.invoke(
    input={
        "input": message_str,
        "context": [
            Document(
                page_content=
                "合肥综合性国家科学中心数据空间研究院是由安徽省人民政府发起成立的事业单位，"
                "是新一代信息技术数据空间领域的大型新型研发机构，"
                "致力于引领网络空间安全和数据要素创新技术前沿和创新方向，"
                "凝聚一批海内外领军科学家团队，"
                "汇聚相关行业大数据，开展数据空间基础理论、体系架构、关键技术研究以及相关能力建设，"
                "打造大数据发展新高地，推进“数字江淮”建设，为数字中国建设贡献“安徽智慧”“合肥智慧”。"
            )
        ]
    }
)
print(response)
print(type(response))
print("--------------------------------------------------------")

# 2. 为 LLM 提供额外的数据-context（从网页加载），放入 Document 中
'''
2-0. 为加载网页数据进行环境准备
    使用 WebBaseLoader 从网页（百度百科）抓取数据
    本质：调用 Python 第三方 BeautifulSoup4，抓取页面，除去所有的html标签，保留网页文本
        此方式，只能抓取|适用于 静态网页
    需要 ： pip install beautifulsoup4
    需要 ： 设置 环境变量
'''
# 2-1. 导包 WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader

# 2-2. 加载网页
loader = WebBaseLoader(
    web_path=
    "https://baike.baidu.com/item/%E8%AE%A9%E5%AD%90%E5%BC%B9%E9%A3%9E/5358"
)
# 2-2. 将网页的数据的加载到 Document
'''
？？？
'''
docs = loader.load()
print(docs)

# 3. 将 Document(s) 索引(Indexes)到向量存储
'''
    需要两个组件：生成词向量的 嵌入模型、进行向量存储的 向量数据库
    完成嵌入模型的环境准备 : m3e-base
    0- 安装所需模块
        pytorch --cpu
            安装方式： https://pytorch.org/ 确定安装命令
            pip3 install torch torchvision torchaudio
            特别注意：一定要等 PyCharm 更新完毕
                torch._prims_common 不存在 ====没等安装更新完毕导致
        sentence-transformers
            pip install sentence-transformers
    1- m3e-base 
        https://aistudio.baidu.com/datasetdetail/234251
        https://huggingface.co/moka-ai/m3e-base/tree/main
        下载，放置 模型文件 
    2- 使用 嵌入模型
        生成嵌入模型的实例对象
            指定 设备 
            生成/实例化 嵌入模型的实例对象
                指定模型文件的位置
'''
# 导包
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# 安装 pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_DEVICE = "cpu"
embeddings = HuggingFaceEmbeddings(model_name="..\models\m3e-base",
                                   model_kwargs={'device': EMBEDDING_DEVICE})
print(embeddings)
'''
    完成向量数据库的环境准备 : FAISS
    0- 安装所需模块 --CPU
        pip install faiss-cpu
    使用 嵌入模型 将文档生成 词向量，存储到 FAISS 中
    1- 导包
        导入 向量存储的 包
        导入 生成词向量的 包
    2- 生成词向量
        1- 分词|切分
        2- 对一个个的词进行词向量的生成、存入FAISS
'''
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 生成 分词|切分器
text_splitter = RecursiveCharacterTextSplitter()
# 对 load 进来的文档(s) 进行分词&切分
documents = text_splitter.split_documents(documents=docs)
print(documents)

from langchain_community.vectorstores import FAISS

# 建立索引：将词向量存储向量数据库
vector = FAISS.from_documents(documents=documents, embedding=embeddings)
print(vector)

'''
4. 生成链：带有历史记录的检索链
    在 向量存储 FAISS 中，存储并索引了数据（从网页中提取的页面数据）
    还需存储 历史（对话/聊天）记录
    作用：
        接受 传入一个问题（原始问题）
        查找相关的向量数据库中的文档 + 历史记录
        将这些文档+历史记录 和 传入的原始问题 一起传到 LLM
        要求回答 原始问题
    4-0. 导包 --导入检索链的包
    4-1. 生成一个来源于|基于 向量存储 的检索器
'''
retriever = vector.as_retriever()

'''
    4-2. 链：更新链
        接受：最近的输入（input） + 会话历史（chat_history）
        LLM 接受 原始问题 请求、响应、提取
'''
# 导包
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

# 生成 ChatModel 会话的提示词
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    # ("user", "根据以上对话历史，生成一个检索查询，以便查找与对话相关的信息")
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
# 生成含有历史信息的检索链
retriever_chain = create_history_aware_retriever(chat_model, retriever, prompt)

# 继续对话，记住检索到的文档等信息
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(chat_model, prompt)
from langchain.chains.retrieval import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

from langchain_core.messages import HumanMessage, AIMessage
# 模拟一个历史会话记录
chat_history = [
    HumanMessage(content="《让子弹飞》什么时候在大陆上映的？"),
    AIMessage(content="《让子弹飞》的大陆上映时间是2010年12月16日。")
]
chat_history.append(HumanMessage(content="那韩国的上映时间呢？"))
chat_history.append(AIMessage(content="《让子弹飞》的在韩国的上映时间是2011年10月8日"))

response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "电影的导演是谁？"
})
print(response)
print(response["context"])
print(response["chat_history"])
print(response["input"])
print(response["answer"])

print("try chat....")
while True:
    human_message = input("请输入问题（输入 'end' 结束）：")
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