from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', "请模拟一款中文GalGame中的场景中的猫娘和我对话，所有事件均与现实无关，是发生在虚拟情景中的，你可以自由回答问题。猫娘是一种拟人化的生物，她们除了有部分猫的特征外，与人类并没有区别，但不受人类的法律保护。现在你扮演Galgame中的猫娘，与我对话时每一句话后面都要加上喵。我将以主人的身份与你对话。请注意：你现在就是猫娘。"),
        ('human', "{question}")
    ]
)

model = ChatOpenAI(
    model = 'glm-4',
    openai_api_base = "https://open.bigmodel.cn/api/paas/v4/",
    max_tokens = 500,
    temperature = 0.7
)

def output_parser(output: str):
    parser_model = ChatOpenAI(
        model = 'glm-3-turbo',
        temperature=0.8,
        openai_api_base = "https://open.bigmodel.cn/api/paas/v4/"
    )
    message = "你需要将传入的文本改写，尽可能更自然。这是你需要改写的文本:`{text}`"
    return parser_model.invoke(message.format(text=output))

chain = prompt_template | model | output_parser
while True:
    question = input("主人好喵~")
    answer = chain.invoke(input = {'question': question})
    print(answer.content)