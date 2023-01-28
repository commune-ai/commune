from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import commune
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# convert the module to a block


llm = HuggingFaceHub(  repo_id="google/flan-t5-xl",  model_kwargs={"temperature":1e-10})
module = commune.module(llm, serve=True)


# serves the module
module.serve()

print(module.serve(wait_for_termination=True, tag='2'))
# # for
# block  = HuggingfaceLLM()
# # block.serve()
# from commune.utils.function import get_functions

# # print(get_functions(HuggingfaceLLM))

# # print the functions of the class
# print(get_functions(HuggingfaceLLM))



