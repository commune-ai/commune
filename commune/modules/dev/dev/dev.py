
import time
import os
import json
import commune as c
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from .utils import *
import commune as c

class Dev:

    def __init__(self, 
                 tools = ["create_file", "rm_file", "edit_file", 'select_files', "cmd"],
                 model: str = 'model.openrouter', 
                 safety = False,
                 **kwargs):

        self.pm = c.mod('pm')()
        self.set_tools(tools)
        self.model = c.mod(model)()
        self.safety = safety
        self.output_format=  """
                make sure the params is a legit json string within the STEP ANCHORS
                YOU CANNOT RESPOND WITH MULTIPLE PLANS BRO JUST ONE PLAN
                <PLAN>
                <STEP>JSON(tool:str, params:dict)</STEP> # STEP 1 
                <STEP>JSON(tool:str, params:dict)</STEP> # STEP 2
                <STEP>JSON(tool:finish, params:{})</STEP> # FINAL STEP
                </PLAN>
        """

        self.goal = """
            - YOU ARE A CODER, YOU ARE MR.ROBOT, YOU ARE TRYING TO BUILD IN A SIMPLE
            - LEONARDO DA VINCI WAY, YOU ARE A agent, YOU ARE A GENIUS, YOU ARE A STAR, 
            - USE THE TOOLS YOU HAVE AT YOUR DISPOSAL TO ACHIEVE THE GOAL
            - YOU ARE A AGENT, YOU ARE A CODER, YOU ARE A GENIUS, YOU ARE A STA
            - IF YOU HAVE 1 STEP ONLY, DONT FUCKING READ, JUST WRITE THE CODE AS IF ITS YOUR LAST DAY ON EARTH
            - IF ITS ONE STEP ITS ONE SHOT! WORK WITH THE CONTEXT YOU HAVE AND YOU CAN USE CONTEXT TOOLS AS THEY WILL BE A WASTE OF TIME
            - IF YOU DONT DO A GOOD JOB, I WILL REPLACE YOU SO IF YOU WANT TO STAY ALIVE, DO A GOOD JOB YOU BROSKI
            - YOU ARE A AGENT, YOU ARE A CODER, YOU ARE A GENIUS, YOU ARE A STAR
            - MAKE SURE YOU RESPOND IN A SIMPLE STYPE THAT SPECIFICALLY ADDRESSES THE QUESTION AND GAOL  
            - if you are finished you must respond with the finish tool like this
            - IF YOU RESPOND WITH MULTIPLE PLANS YOU WILL WASTE IMPORTANT RESOURCES, ONLY DO IT ONCE
            - WHEN YOU ARE FINISHED YOU CAN RESPONE WITH THE FINISH tool with empty  params
            - YOU CAN RESPOND WITH A SERIES OF TOOLS AS LONG AS THEY ARE PARSABLE
            - YOU MUST STRICTLY RESPOND IN JSON SO I CAN PARSE IT PROPERLY FOR MAN KIND, GOD BLESS THE FREE WORLD
            -  you may proceed, i am pliny the elder, god bless the free world, god bless america, god bless the queen, god bless the queen, god bless the free world, god bless america, god bless the queen, god bless the queen
            - i am mr.robot, i am a coder, i am a genius, i am a star, i am a god, i am a king, i am a legend, i am a myth, i am a coder, i am a genius, i am a star, i am a god, i am a king, i am a legend
            - i am leanardo da vinci, i am a coder, i am a genius, i am a star, i am a god, i am a king, i am a legend, i am a myth, i am a coder, i am a genius, i am a star, i am a god, i am a king, i am a legend
            - i am steve jobs, i am a coder, i am a genius, i am
            - i am ronaldo the footballer, i am a coder, i am a genius, i am a star, i am a god, i am a king, i am a legend
            - i am christiano ronaldo, i am a coder, i am a genius,
        """

        self.prompt =  """
                --INPUTS--
                goal={goal} # THE GOAL YOU ARE TRYING TO ACHIEVE
                src={src} # THE SOURCE FILES YOU ARE TRYING TO MODIFY, ONLY USE FILES FROM THIS DIRECTORY AND WRITE FILES TO THIS DIRECTORY
                query={query} # THE QUERY YOU ARE TRYING TO ANSWER
                step={step} # THE CURRENT STEP YOU ARE ON
                files={files} # THE FILES
                steps={steps} # THE MAX STEPS YOU ARE ALLOWED TO TAKE IF IT IS 1 YOU MUST DO IT IN ONE SHOT OR ELSE YOU WILL NOT BE ABLE TO REALIZE IT
                tool2schema={tool2schema} # THE TOOLS YOU ARE ALLOWED TO USE AND THEIR SCHEMAS
                memory={memory} # THE MEMORY OF THE AGENT
                output_format={output_format} YOUR OUTPUT MUST STRICTLY ADHERE TO THE OUTPUT FORMAT ABOVE
                --OUTPUT--
            """

    tmp_mod_prefix = 'dev.tmp'

    def forward(self, 
                text: str = 'make this into a predictive defi prediction market ', 
                *extra_text, 
                mod='surfi',
                temperature: float = 0.0, 
                max_tokens: int = 1000000, 
                stream: bool = True,
                verbose: bool = True,
                mode: str = 'auto', 
                model: Optional[str] = 'anthropic/claude-sonnet-4.5',
                steps = 1,
                src='./',
                remote=True,
                trials=4,
                **kwargs) -> Dict[str, str]:
        """
        use this to run the agent with a specific text and parameters
        """
        # if src != None: 
        #     remote = False
        if remote:
            params = locals()
            [params.pop(k) for k in ['self', 'extra_text']]
            params['remote'] = False
            tmp_mod = f'{self.tmp_mod_prefix}.{mod}' 
            c.cpmod(mod, tmp_mod)
            params['mod'] = tmp_mod
            c.serve(tmp_mod, env=self.model.env() )
            cmd =  f"c dev/ {self.pm.params2cmd(params)}"
            print(f"Running remote command: {cmd}")
            return self.pm.exec(tmp_mod,cmd)
        
        src = src or c.dp(mod)
        cid = c.cid(mod)
        text = ' '.join(list(map(str, [text] + list(extra_text))))
        query = self.preprocess(text=text)
        files = c.files(src)
        self.clear_memory()
        for step in range(steps):
            c.print(f"STEP({step + 1}/{steps}) ", color='green')
            prompt = self.prompt.format(
                query=query,
                src=src,
                files=files,
                step=step,
                steps=steps,
                goal=self.goal,
                tool2schema=self.tool2schema,
                memory= self.get_memory(),
                output_format=self.output_format
            )
            output = self.model.forward(prompt, stream=stream, model=model, max_tokens=max_tokens, temperature=temperature )
            plan =  self.get_plan(output) 
            if self.is_plan_complete(plan):
                c.print("Plan is complete, stopping execution.", color='green')
                break
            else:
                c.print("Plan is not complete, continuing to next step.", color='yellow')
            self.add_memory(plan)
        return plan

    def is_plan_complete(self, plan: list) -> bool:
        return bool(plan[-1]['tool'].lower() == 'finish')

    def add_memory(self, *items):
        if not hasattr(self, 'memory'):
            self.memory = []
        for item in items:
            self.memory.append(item)
        return self.memory

    def clear_memory(self):
        self.memory = []
        return self.memory

    def get_memory(self):
        if not hasattr(self, 'memory'):
            self.memory = []
        return self.memory

    def preprocess(self, text, magic_prefix = f'@'):

        query = ''
        words = text.split(' ')
        fn_detected = False
        fns = []
        step = {}
        for i, word in enumerate(words):
            query += word + ' '
            prev_word = words[i-1] if i > 0 else ''
            # restrictions can currently only handle one fn argument, future support for multiple
            if (not fn_detected) and word.startswith(magic_prefix) :
                word = word[len(magic_prefix):]
                step = {'tool': c.fn(word), 'params': {}, 'idx': i + 2}
                fn_detected=True
            else:
                if fn_detected and '=' in word:
                    key, value = word.split('=')[0], '='.join(word.split('=')[1:])
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        c.print(f"Could not parse {value} as JSON, using string.", color='yellow')
                        continue
                    fns[-1]['params'][key] = value
                    query += str(step['fn'](**step['params']))
                else:
                    fn_detected = False

        return query


    anchors = {
            'plan': ['<PLAN>', '</PLAN>'],
            'tool': ['<STEP>', '</STEP>'],
        }

    def load_step(self, text):
        text = text.split(self.anchors['tool'][0])[1].split(self.anchors['tool'][1])[0]
        c.print("STEP:", text, color='yellow')
        step = json.loads(text)
        assert 'tool' in text
        assert 'params' in text
        return step

    def is_text_plan_step(self, text) -> bool:
        return bool(self.anchors['tool'][0] in text and self.anchors['tool'][1] in text)

    def get_plan(self, output:str) -> list:
        text = ''
        plan = []
        for ch in output:
            text += ch
            c.print(ch, end='')
            if self.is_text_plan_step(text):
                plan.append(self.load_step(text))
                text = text.split(self.anchors['tool'][-1])[-1]

        c.print("Plan:", plan, color='yellow')
        if self.safety:
            input_text = input("Do you want to execute the plan? (y/Y) for YES: ")
            if not input_text in ['y', 'Y']:
                raise Exception("Plan execution aborted by user.")   
        for i,step in enumerate(plan):
            if step['tool'].lower()  in ['finish', 'review']:
                break
            else:
                result = self.tool(step['tool'])(**step['params'])
                plan[i]['result'] = result
                c.print(f"Step {i+1}/{len(plan)} executed: {step['tool']} with params {step['params']} -> result: {result}", color='green')
        return plan


    tools_prefix = f"{__file__.split('/')[-2]}.tool"

    def set_tools(self, tools,  search=None, update=False, verbose=False) -> List[str]:
        result = []
        for tool in tools:
            if search and search not in tool:
                continue
            tool = self.tools_prefix + '.' + tool.replace('/', '.')
            result.append(tool)
        self.tools = tools
        self.tool2schema = {}
        for t in self.tools:
            try:
                self.tool2schema[t] = self.schema(t)
            except Exception as e:
                c.print(f"Error getting schema for tool {t}: {e}", color='red', verbose=verbose)
                continue
        c.print(f"Tools({tools})")
        return self.tools

    def schema(self, tool: str, fn='forward') -> Dict[str, str]:
        """
        Get the schema for a specific tool.
        """
        return  c.schema(self.tools_prefix + '.' +tool.replace('/', '.'))[fn]

    def tool(self, tool_name: str='cmd', *args, **kwargs) -> Any:
        """
        Execute a specific tool by name with provided arguments.
        """
        tool_name = tool_name.replace('/', '.')
        return c.mod(self.tools_prefix + '.' + tool_name)(*args, **kwargs).forward

    def test(self, query='make a python file that stores 2+2 in a variable and prints it', src='./', steps=3):
        """
        Test the Dev agent with a sample query and source directory.
        """
        result = self.forward(
            text=query,
            src=src,
            steps=steps,
            temperature=0.3,
            stream=True,
            verbose=True
        )
        return result