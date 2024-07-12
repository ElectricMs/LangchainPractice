# 初始化 AKs&SKs
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
chat_model=spark_chat_model
#chat_model=zhipu_chat_model

#from langchain.schema import (AIMessage,SystemMessage,HumanMessage)
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

"""
以下是一些可能在 langchain.schema 中找到的关键类和概念：

Message：表示单个消息的数据模型，通常包含文本内容、发送者信息等。
BaseMessage：消息的基类，提供所有消息类型共有的属性和方法。
HumanMessage：专门表示人类用户发送的消息。
AIMessage：专门表示 AI 或聊天机器人发送的消息。
SystemMessage：用于传达系统级指令或信息的消息。
PromptTemplate：用于定义如何生成提示（prompts）的模板，这些提示将用于与语言模型交互。
BasePromptTemplate：提示模板的基类，提供了模板的基本结构和行为。
HumanMessagePromptTemplate：用于生成人类用户消息的提示模板。
Chain：表示处理输入和生成输出的操作序列，是 langchain 中实现逻辑和工作流的核心概念。
BaseChain：链的基类，定义了链的基本接口和执行逻辑。

这些类和结构为 langchain 提供了一个标准化的框架，使得开发者能够更容易地构建和组合不同的语言模型应用。
使用 langchain.schema，开发者可以定义如何处理输入、如何与模型交互以及如何解释模型的输出。
"""

#1-1.生成单一的message 因为是chatmodel，所以是message list
if(0):
    human_msgs=[HumanMessage(content="hi")]
    response=chat_model.invoke(input=human_msgs)
    print(response)



#1-2.生成多个message
if(0):
    system_human_msgs=[SystemMessage(content="you are a student"),HumanMessage(content="who are you?")]
    response=chat_model.invoke(input=system_human_msgs)
    print(response)

#1-3.生成多轮多个message
if(0):
    print("1-3.生成多轮多个message-------------------------")
    batch_msgs=[
        [SystemMessage(content="you are a student"),HumanMessage(content="who are you?")],
        [SystemMessage(content="you are a teacher"),HumanMessage(content="who are you?")]
    ]
    result = chat_model.generate(batch_msgs)
    print(result)
    #print(type(result))
    print("ans1:"+result.generations[0][0].message.content)
    #print("ans1.token_usage:"+result.generations[0][0].message.response_metadata['token_usage'])
    #字典还不能直接打印


#2. prompt template
from langchain.prompts.chat import (
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


#2-1. 生成模板字符串，带有占位符
system_template_str = "你是一名专业的翻译家，擅长将 {input_language} 翻译为 {output_language}."
human_template_str = "{text}"
#2-2. 生成提示词 基于模板生成
system_message_prompt=SystemMessagePromptTemplate.from_template(system_template_str)
human_message_prompt=HumanMessagePromptTemplate.from_template(human_template_str)
#2-3. 创建聊天提示模板
chat_prompt=ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])

if(0):
    chat_msgs=chat_prompt.format_messages(input_language="英语", output_language="中文", text="I love programming.")
    print(chat_msgs)
    response=chat_model.invoke(input=chat_msgs)
    print(response.content)

#3. chain=LLMChain
if(0):
    from langchain import LLMChain
    llm_chain=LLMChain(
        llm=chat_model,
        prompt=chat_prompt
    )
    response=llm_chain.invoke(
        input={
            "input_language": "英语",
            "output_language": "中文",
            "text": "I love programming."
        }
    )
    print(response)

#4. Agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import (initialize_agent, AgentType)
"""
Agent进行网络资源搜索|检索，实现方式:
1．给出第三方接口的函数（工具）的名称:serpapi
2．封装爬虫工真，爬取指定页面(s)
3. 自行实现一个调用某个搜索引擎（发关键词获取并解析搜索反馈）函数
"""
# https://blog.csdn.net/wenxingchen/article/details/130474611
# 获取 serapi google搜索工具的key



tools=load_tools(tool_names=["serpapi","llm-math"],llm=chat_model)

agent=initialize_agent(
    tools=tools,
    llm=chat_model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_error=True,#处理运行时错误，不影响程序运行
)

agent.run("谁是鲁迅的妻子？她哪一年逝世？")
































exit()

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 定义系统消息模板
system_template = "你是一名专业的翻译家，擅长将 {input_language} 翻译为 {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
# 定义人类消息模板
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# 创建聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])#传入list
# 格式化消息
message = chat_prompt.format_messages(input_language="英语", output_language="中文", text="I love programming.")
print(message)

response = chat_model.invoke(input=message)
print(response.content)