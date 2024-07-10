'''
0. 导包
1. 获取 AK&SK
2. 实例一个langchain包装的wenxin模型
3. 发送对话、获取响应、输出响应内容
'''
# from langchain_wenxin import Wenxin
from langchain_wenxin.llms import Wenxin
from dotenv import load_dotenv
load_dotenv()

llm = Wenxin(
    temperature = 0.9,
    model="ernie-bot-turbo",    #注意小写！！
    baidu_api_key=API_KEY,
    baidu_secret_key=SECRET_KEY,
    verbose=True,
)

response = llm("hello")
print(response)

response = llm("你的版本？")
print(response)