import commune as c
import openai
import time
# import asyncio

class ModelGPTs(c.Module):
    def __init__(self, api_key: str = None, **kwargs):
        self.set_config(locals())
        api_key = api_key if api_key != None else self.get_api_key()
        self.gpt_client = openai.Client(api_key=api_key)

        self.tutors = {}
        self.tutors['math'] = self.gpt_client.beta.assistants.create(
            name="Math Tutor",
            instructions="You are a personal math tutor. Write and run code to answer math questions",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-1106-preview"
        )

    ## ! API MANAGEMENT ##

    @classmethod
    def set_api_key(self, api_key:str, cache:bool = True):
        if api_key == None:
            api_key = self.get_api_key()


        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)

    @classmethod
    def add_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}
    
    @classmethod
    def rm_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   

        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def get_api_key(cls):
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return c.choice(api_keys)

    @classmethod
    def api_keys(cls):
        return cls.get('api_keys', [])
    
    ## API MANAGEMENT ! ##

    def call(self,
             type: str, # one of ('math')
             content: str # the content to be processed
    ) -> str:
        assert type in self.tutors.keys(), "Invalid type. Must be one of {}".format(self.tutors.keys())

        thread = self.gpt_client.beta.threads.create()
        message = self.gpt_client.beta.threads.messages.create(
            thread_id=thread.id,
            role='user',
            content=content
        )

        run = self.gpt_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.tutors[type].id,
            instructions="Please address the user as John Doe. The user has a premium account."
        )

        for _ in range(10):
            """
            This loop is necessary because the API is not synchronous.
            Retry upto 10 times to get the result.
            """
            complete_run = self.gpt_client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if complete_run.status == 'completed':
                messages = self.gpt_client.beta.threads.messages.list(thread_id=thread.id)
                return messages
            time.sleep(2)

        # async def _async_retrieve(**kwargs):
        #     self.gpt_client.beta.threads.runs.retrieve(**kwargs)

        # async def _async_list(**kwargs):
        #     self.gpt_client.beta.threads.messages.list(**kwargs)

        # asyncio.run(_async_retrieve(thread_id=thread.id, run_id=run.id))

        # messages = asyncio.run(_async_list(thread_id=thread.id))

        return "Error: Could not retrieve messages."