from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

template_1= PromptTemplate(template="""what is the capital city of {country}""",
                           input_variables=['country'])

prompt1= template_1.invoke({'country':'India'})


llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen3.5-9B",task="text-generation")

model=ChatHuggingFace(llm=llm)

result=model.invoke(prompt1)

template_2=PromptTemplate(template="""give me information about the following city : {city} in 1-2 lines""",
                          input_variables=['city'])



prompt2=template_2.invoke({'city':result.content})

result2=model.invoke(prompt2)

print(result)
print("=====================================================")
print(result2)

print("=====================================================")
print(prompt1)
print("=====================================================")
print(prompt2)