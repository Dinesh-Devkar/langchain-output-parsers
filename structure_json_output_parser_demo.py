from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

response_schema= [
    ResponseSchema(name="first_fact",description="first fact about the topic"),
    ResponseSchema(name = 'second_fact',description='second fact about the topic')
]

parser=StructuredOutputParser.from_response_schemas(response_schema)

template=PromptTemplate(template="give me two facts about the {topic}. \n {format_instructions}",
                        input_variables=['topic'],
                        partial_variables={'format_instructions':parser.get_format_instructions()})

prompt=template.invoke({'topic':'Earth'})

model=ChatAnthropic(model_name='claude-haiku-4-5')

result=model.invoke(prompt)

print(result)
print("="*50)

final_result=parser.parse(result.content)

print(final_result)
