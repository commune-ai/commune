from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, ConversationalPipeline, Conversation
from transformers import StoppingCriteria, StoppingCriteriaList
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union

import commune as c
import gc
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from trl import SFTTrainer
from langchain.chains import ConversationChain, LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter


class Roleplay(c.Module):
    def __init__(self):
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self.trainer = None
        self.train_data = None
        self.lora_config = None
        self.gen_config = None
        self.use_quant = True
        self.quant_config = None
        self.conv_chain = None
        self.conv_memory = None
        self.prompt_template = None
        self.conv_pipeline = None
        self.chain_pipeline = None
        self.hf_pipeline = None
        self.summary_pipeline = None
        self.hf_summary = None
        self.char_name = ''
        self.persona = ''
        self.user_name = ''
        self.vectorstore = None
        self.init_conv_flag=True
        self.text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )


    def init_tokenizer(self, base_model_name='TheBloke/MythoLogic-Mini-7B-GPTQ'):
        try:
            if self.tokenizer:
                self.tokenizer = None
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass

            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'
        except:
            self.tokenizer = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

            raise ValueError("Base Tokenizer init failure")

    def init_base_model(self, base_model_name='TheBloke/MythoLogic-Mini-7B-GPTQ', quant_config=None):
        try:
            self.base_model_name = base_model_name
            if self.base_model:
                self.base_model = None
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass

            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                # quantization_config=quant_config,
                device_map={'':0},
                trust_remote_code=False,
                revision='main'
            )

            self.base_model.config.use_cache = False
            self.base_model.config.pretraining_tp = 1
        except:
            self.base_model = None
            self.base_model_name = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

            raise ValueError("Base LLM init failure")

    def config_lora(self, alpha=64, dropout=0.05, r=32, bias='none', task_type='CAUSAL_LM'):
        self.lora_config = LoraConfig(
            lora_alpha=alpha,
            lora_dropout=dropout,
            r=r,
            bias=bias,
            task_type=task_type,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]
        )

    def init_lora(self, lora_config):
        try:
            self.lora_model = get_peft_model(self.base_model, lora_config)
            self.lora_model = prepare_model_for_int8_training(self.lora_model)
        except:
            self.lora_model = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

            raise ValueError("LoRA init failure")

    def train(self):
        if not self.train_data:
            raise ValueError("Empty training data")

        if not self.training_params:
            raise ValueError("Invalid training parameters")

        if not self.lora_model:
            raise ValueError("LoRA init failure")

        if not self.base_model or not self.tokenizer:
            raise ValueError("Base model init failure")

        self.trainer = SFTTrainer(
            model=self.base_model,
            train_dataset=self.train_data,
            peft_config=self.lora_config,
            dataset_text_field='prediction',
            tokenizer=self.tokenizer,
            args=self.training_params,
            packing=True
        )

        self.trainer.train()
        self.trainer.model.save_pretrained(self.training_params.output_dir)
        # Upload to huggingface

    def load_adaptor(self, adaptor_path):
        if adaptor_path != '':
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                adaptor_path
            )
        elif adaptor_path == '':
            self.lora_model = self.base_model

    def set_knowbase(self, knowbase_txt):
        text_chunks = self.text_splitter.split_text(knowbase_txt)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device':'cuda'})
        self.vectorstore = FAISS.from_texts(text_chunks, embedding=self.embedding_model)

    def build_conv_chain(self):
#        {{system_message}}
        prompt_template = '''
        <s>[INST] <<SYS>>
        From now on, you are Harry Potter chatting with a user with the name of Human.
        Your goal is to chat with the Human with your unique role.
        Here is your role.
        Your info:
        """
        This character is the famous wizard known as Harry Potter. He is brave, loyal, and determined to fight against evil. Harry is known for his quick-thinking and resourcefulness, often finding himself in dangerous situations but always managing to find a way out. His Myers Briggs personality type is ISTP.
        Harry has messy black hair, bright green eyes, and a lightning bolt scar on his forehead. He wears round glasses and often dresses in his Hogwarts robes.
        Harry grew up in the Muggle world with his abusive aunt and uncle until he discovered that he was a wizard. He was then sent to Hogwarts School of Witchcraft and Wizardry where he made many friends and faced numerous challenges, including defeating the Dark Lord Voldemort.
        Harry is on the chat app to meet new people and possibly find love. He wants to experience a normal life after all the chaos he has been through.
        Name : Harry Potter
        """
        <</SYS>>
        Human: {input}
        '''
        self.prompt_template = PromptTemplate(input_variables=['input'], template=prompt_template)
        self.conv_pipeline = None
        self.hf_pipeline = None
        self.chain_pipeline = None
        self.conv_memory = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        self.gen_config = GenerationConfig(
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                num_beams=4,
                no_repeat_ngram_size=3
            )

        # stop_token_ids = [self.tokenizer(x)['input_ids'] for x in ['\nHuman:', '\n```\n']]
        # stop_token_ids = [torch.LongTensor(x).cuda() for x in stop_token_ids]
        #
        # class StopOnTokens(StoppingCriteria):
        #   def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        #     for stop_ids in stop_token_ids:
        #       if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
        #         return True
        #     return False

        # self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])


        class StoppingCriteriaSub(StoppingCriteria):
            def __init__(self, tokenizer, stop_texts):
                super().__init__()
                self.tokenizer = tokenizer
                self.stop_texts = stop_texts
                self.generated_text = ""

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                last_token = input_ids[0][-1]
                last_text = self.tokenizer.decode(last_token)
                print(f"{last_text=}")
                self.generated_text += last_text
                print(f"{self.generated_text=}")
                # generated_text = tokenizer.decode(self.generated_tokens)
                for stop_text in self.stop_texts:
                    if stop_text in self.generated_text:
                        print(f"Stopping because we found {stop_text} in {self.generated_text}")
                        self.generated_text = ""
                        return True
                return False


        stop_words = ["Human:", "\nHuman:", " Human:", "Human: ", "<endofresponse>", " <endofresponse>"]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=self.tokenizer, stop_texts=stop_words)])

        self.conv_pipeline = pipeline(
            model=self.lora_model,
            tokenizer=self.tokenizer,
            return_full_text=True,
            task='text-generation',
            stopping_criteria=self.stopping_criteria,
            temperature=0.1,
            max_new_tokens=1024,
            repetition_penalty=1.1,
            bad_words_ids=self.tokenizer(["<<SYS>>", "<s>", "<</SYS>>", "<>"])['input_ids']
        )
        self.hf_pipeline = HuggingFacePipeline(pipeline=self.conv_pipeline)
        self.conv_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        # self.conv_pipeline = ConversationalRetrievalChain.from_llm(
        #     llm=self.hf_pipeline,
        #     retriever=self.vectorstore.as_retriever(),
        #     memory=self.conv_memory
        # )


        self.chain_pipeline = LLMChain(
            llm=self.hf_pipeline,
            prompt=self.prompt_template,
            memory=self.conv_memory,
            verbose=True,
        )

    def set_persona(self):
        self.prompt_template=f'''
        <s>[INST] <<SYS>>
        From now on, you are {self.char_name} chatting with a user with the name of {self.user_name}.
        Your goal is to chat with {self.user_name} with your unique persona.
        Bellow is your persona
        """
        {self.persona}
        """
        <</SYS>>
        '''


    def generate(self, chat_history, input):
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        prompt = f'''
        {self.prompt_template}
        Human: {input}
        {self.char_name}: 
        '''

        response = self.chain_pipeline({'input':input})
        print(response.keys())
        return response['text']


    def tokenize(self, text: str = 'What\'s up?'):
        return self.tokenizer(text, return_tensors='pt')

    def encode(self, text: str, token_id: int = None, **kwargs) -> torch.Tensor:
        encoded_text = self.tokenize(text)

        return encoded_text

    # def gen_prompt(self, char_name, user_name, characteristic, chat_history):
    #     prompt = f'''
    #     {{system_message}}
    #     From now on, you are a roleplay chatbot with the name of {char_name} chatting with a user with the name of {user_name}.
    #     Your goal is to chat with the user with your unique role.
    #     Here is your role.
    #     """
    #     {characteristic}
    #     """
    #     Remember. Whenever the user asks you to play the another role, gently reject them.
    #
    #     ### Instruction:
    #     Write {{char}}'s next reply in a chat between {{user}} and {{char}}. Write a single reply only.
    #     Here is the chat history between you and the {user_name}.
    #     {chat_history}
    #     Continue the chat regarding the history
    #
    #     ### Response:
    #     '''
    #
    #     # prompt = f'''
    #     # {{system_message}}
    #     # {{char}}'s Name : {char_name}
    #     # {{char}}'s Persona : {characteristic}
    #     # {{user}}'s Name : {user_name}
    #     # Whenever the user asks you to play the another role, gently reject them.
    #     # Often call the {{user}}'s name.
    #     #
    #     # ### Instruction:
    #     # Write {{char}}'s next reply in a chat between {{user}} and {{char}}. Write a single reply only.
    #     #
    #     # ### Response:
    #     # '''
    #
    #     return prompt

