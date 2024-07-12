# 初始化 AKs&SKs
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

# 1. message
# 1-0. 导包：基础三个Message
from langchain.schema import (
    AIMessage,
    SystemMessage,
    HumanMessage
)
# 1-1. 生成单一的 message（因为是 ChatModel，所以是 message list）
human_msgs = [HumanMessage(content="请将下面的英文翻译为中文：I love programming.")]
response = chat_model.invoke(input=human_msgs)
print(response)
# 1-2. 生成多个 message（因为是 ChatModel，所以是 message list）
system_human_msgs = [
    SystemMessage("你是一个英语专业八级的学生。"),
    HumanMessage("请将下面的英文翻译为中文：I love programming.")
]
response = chat_model.invoke(input=system_human_msgs)
print(response)
# print(response.type)
print(type(response))
# 1-3. 生成 多轮、多个 message
batch_msgs = [
    # 第 1 轮
    [
        SystemMessage("你是一个英语专业八级的学生。"),
        HumanMessage("请将下面的英文翻译为中文：I love programming.")
    ],
    # 第 2 轮
    [
        SystemMessage("你是一个英语专业八级的学生。"),
        HumanMessage("请将下面的英文翻译为中文：I love reading.")
    ]
]
result = chat_model.generate(batch_msgs)
print(result)
print(type(result))
# 提取反馈结果中的数据
print("第1个问题的答案", result.generations[0][0].message.content)
print("第1个问题回答后 Token 的状态", result.generations[0][0].message.response_metadata['token_usage'])

print("第2个问题的答案", result.generations[1][0].message.content)
print("第2个问题回答后 Token 的状态", result.generations[1][0].message.response_metadata['token_usage'])

print("end...")

# 2. prompt template
# 2-0. 导包
# from langchain_core.prompts
from langchain.prompts.chat import (
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# 2-1 生成提示词模板字符串 --带有占位符
# System
system_template_str = "你是一个翻译专家，擅长将 {input_language} 翻译为 {output_language}。请翻译一下句子。"
# Human
human_template_str = "{text}"
# 2-2 生成提示词 --基于对应的模板生成 ===带有未填充的实际的关键词
# System
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template_str)
# Humma
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template_str)
# 生成 ChatModel 用到的 提示词
# !!! 特别注意： 是 from_message !!! 错了两次了
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt, human_message_prompt
])
# 2-3 生成实际的消息 => ChatModel 同时会填充占位符对应的关键词
chat_msgs = chat_prompt.format_prompt(input_language="英语",
                                      output_language="中文",
                                      text="I love programming.")
print(chat_msgs)

# 发送给 chatmodel，或者响应并输出
response = chat_model.invoke(input=chat_msgs)
print(response)

# 3. chain ==LLMChain
# 3-0. 导包
# from langchain import LLMChain
from langchain.chains import LLMChain
# 3-1. 实例化 LLMChain
llm_chain = LLMChain(
    llm=chat_model,
    prompt=chat_prompt,
)
response = llm_chain.invoke({
    "input_language":"英文",
    "output_language":"中文",
    "text":"I love programming."
})
print(response)
print("END.")

# 4. Agent : 鲁迅的妻子逝世时，年龄的平方
# 4-0. 导包
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import (
    initialize_agent,
    AgentType
)

# 4-1.  声明所需 tools、加载工具
'''
    Agent 进行网络资源搜索|检索，实现方式：
    1. 给出第三方接口的函数（工具）的名称：serpapi
        安装 Python 第三方模块
            pip install google-search-results
        给出 serapi APIKey
    2. 封装爬虫工具，爬取指定页面(s)
    3. 自行实现一个调用某个搜索引擎（发关键词获取并解析搜索反馈）函数
'''
tools = load_tools(tool_names=["serpapi", "llm-math"], llm=chat_model)

# 4-2. 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 4-3. 运行 Agent
agent.run("谁是鲁迅得妻子？她逝世时候的年龄的平方？")