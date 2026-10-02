from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatAnthropic(model_name="claude-haiku-4-5")

template1=PromptTemplate(template="""what is the capital city of {country}""",
                         input_variables=['country'])

prompt1=template1.invoke({'country':'Nepal'})

result1=model.invoke(prompt1)

template2=PromptTemplate(template="""give me information about the following city : {city} in 1-2 lines.""",
                         input_variables=['city'])

prompt2=template2.invoke({'city':result1.content})

result2=model.invoke(prompt2)

print(result1.content)
print("=================================================================")
print(result2.content)

str_parser=StrOutputParser()

chain= template1 | model | str_parser | template2 | model | str_parser

final_result= chain.invoke({'country':'India'})

print(final_result)

