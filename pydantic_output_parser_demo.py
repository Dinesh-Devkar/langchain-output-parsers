from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Annotated,Optional

load_dotenv()

class Person(BaseModel):
    name: str= Field(description='name of the person')
    age:int = Field(gt=18,description='age of the person')
    hobbies:Optional[list[str]]= Field(description='hobbies of the person')

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(template=""" give me name,age and hobbies of a fictional person of country {country}. \n {format_instructions}""",
                        input_variables=['country'],
                        partial_variables={'format_instructions':parser.get_format_instructions()})

prompt=template.invoke({'country':'India'})

print(prompt)

print("="*30)

model=ChatAnthropic(model_name='claude-haiku-4-5')

result=model.invoke(prompt)

final_result=parser.parse(result.content)

print(final_result)