from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import commune
import os


os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# convert the module to a block



class HuggingfaceLLM(commune.Module):
    def __init__(self):
        self.llm = HuggingFaceHub(  repo_id="google/flan-t5-xl",  model_kwargs={"temperature":1e-10})
        self.module = commune.module(llm, serve=True)


if __name__ == "__main__":
    HuggingfaceLLM.run()