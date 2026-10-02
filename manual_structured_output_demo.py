from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated,Optional
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class Country(BaseModel):
    capital:Optional[Annotated[str,'Capital city of requested country']]=None

my_country=Country()

llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3.5-9B',task='text-generation')

model=ChatHuggingFace(llm=llm)

# model=model.with_structured_output(Country, method="json_schema")

result=model.invoke('what is the capital of Bhutan')

parser= StrOutputParser()

parser_result=parser.invoke(result)

print(result)
print("===============================================")
print(parser_result)