from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv


load_dotenv()

model=ChatAnthropic(model_name='claude-haiku-4-5')

parser=JsonOutputParser()

template=PromptTemplate(template="""give me five facts about the {topic}. \n {format_instructions}""",
                        input_variables=['topic'],
                        partial_variables={'format_instructions':parser.get_format_instructions()})

# prompt=template.format(topic='india')

prompt= template.invoke({'topic':'bhutan'})

print(prompt)
print("===============================================================")

result=model.invoke(prompt)

# print(result.content)

# final_result=parser.parse(result.content)

# print(final_result)
# print("===============================================================")
# print(type(final_result))


# Calling Via Chain

# chain= template | model | parser

# res=chain.invoke({'topic':'japan'})

# print(res)
# print(type(res))