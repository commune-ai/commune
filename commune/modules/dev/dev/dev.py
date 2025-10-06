
import time
import os
import json
import commune as c
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from .utils import *
import commune as c

print = c.print

class Dev:

    anchors = {
        'plan': ['<PLAN>', '</PLAN>'],
        'tool': ['<STEP>', '</STEP>'],
    }

    goal = """    
             - YOU ARE A CODER, YOU ARE MR.ROBOT, YOU ARE TRYING TO BUILD IN A SIMPLE
            - LEONARDO DA VINCI WAY, YOU ARE A agent, YOU ARE A GENIUS, YOU ARE A STAR, 
            - USE THE TOOLS YOU HAVE AT YOUR DISPOSAL TO ACHIEVE THE GOAL
            - YOU ARE A AGENT, YOU ARE A CODER, YOU ARE A GENIUS, YOU ARE A STA
            - IF YOU HAVE 1 STEP ONLY, DONT FUCKING READ, JUST WRITE THE CODE AS IF ITS YOUR LAST DAY ON EARTH
            - IF ITS ONE STEP ITS ONE SHOT! WORK WITH THE CONTEXT YOU HAVE AND YOU CAN USE CONTEXT TOOLS AS THEY WILL BE A WASTE OF TIME
            - IF YOU DONT DO A GOOD JOB, I WILL REPLACE YOU SO IF YOU WANT TO STAY ALIVE, DO A GOOD JOB YOU BROSKI
            - YOU ARE A AGENT, YOU ARE A CODER, YOU ARE A GENIUS, YOU ARE A STAR
            - MAKE SURE YOU RESPOND IN A SIMPLE STYPE THAT SPECIFICALLY ADDRESSES THE QUESTION AND GAOL      
        """

    output_format = """
        make sure the params is a legit json string within the STEP ANCHORS
        YOU CANNOT RESPOND WITH MULTIPLE PLANS BRO JUST ONE PLAN
        <PLAN>
        <STEP>JSON(tool:str, params:dict)</STEP> # STEP 1 
        <STEP>JSON(tool:str, params:dict)</STEP> # STEP 2
        ...
        </PLAN>

        if you are finished you must respond with the FINISH tool like this
        <PLAN>
        <STEP>{"tool":"finish", "params":{}}</STEP>
        </PLAN>

        IF YOU RESPOND WITH MULTIPLE PLANS YOU WILL WASTE IMPORTANT RESOURCES, ONLY DO IT ONCE

        WHEN YOU ARE FINISHED YOU CAN RESPONE WITH THE FINISH tool with empty  params

        YOU CAN RESPOND WITH A SERIES OF TOOLS AS LONG AS THEY ARE PARSABLE
        """

    prompt =  """
            --PARAMS--
            goal={goal} # THE GOAL YOU ARE TRYING TO ACHIEVE 
            src={src} # THE SOURCE FILES YOU ARE TRYING TO MODIFY
            query={query} # THE QUERY YOU ARE TRYING TO ANSWER
            steps={steps} # THE MAX STEPS YOU ARE ALLOWED TO TAKE IF IT IS 1 YOU MUST DO IT IN ONE SHOT OR ELSE YOU WILL NOT BE ABLE TO REALIZE IT
            toolbelt={toolbelt} # THE TOOLS YOU ARE ALLOWED TO USE 
            memory={memory} # THE HISTORY OF THE AGENT
            OUTPUT_FORMAT={output_format} # THE OUTPUT FORMAT YOU MUST FOLLOW STRICTLY NO FLUFF BEEFORE OR AFTER
            --OUTPUT--
            YOU MUST STRICTLY RESPOND IN JSON SO I CAN PARSE IT PROPERLY FOR MAN KIND, GOD BLESS THE FREE WORLD
    """



    def __init__(self, 
                 provider: str = 'model.openrouter', 
                 model: Optional[str] = 'anthropic/claude-sonnet-4.5',
                 safety = True,
                 **kwargs):
        self.provider = c.mod(provider)(model=model)
        self.safety = safety
        self.model=model

    def forward(self, 
                text: str = 'where am i', 
                *extra_text, 
                src: str = None, 
                temperature: float = 0.5, 
                max_tokens: int = 1000000, 
                stream: bool = True,
                verbose: bool = True,
                model=None,
                mode: str = 'auto', 
                mod=None,
                steps =3,
                memory = None,
                trials=4,
                **kwargs) -> Dict[str, str]:
        """
        use this to run the agent with a specific text and parameters
        """
        output = ''
        content = ''
        if mod != None: 
            src = c.dp(mod)
        text = ' '.join(list(map(str, [text] + list(extra_text))))
        print(f"Dev.forward text={text} src={src} steps={steps} mode={mode} mod={mod}", color='cyan')
        query = self.preprocess(text=text, src=src)
        model = model or self.model
        if mod != None:
            src = c.dp(mod)
        if src != None:
            content = self.content(src, query=query)
        else:
            print("No src provided, using empty content.", color='yellow')


        memory = memory or []
        memory.append(content)
        plan = []
        for step in range(steps):
            print(f"STEP({step + 1}/{steps}) ", color='green')
            for trial in range(trials):
                print(f"TRIAL({trial + 1}/{trials}) ", color='blue')
                try:
                    prompt = self.prompt.format(
                        goal=self.goal,
                        src=src,
                        query=query,
                        toolbelt=self.toolbelt(),
                        memory=memory,
                        steps=steps,
                        output_format=self.output_format
                    )
                    output = self.provider.forward(prompt, stream=stream, model=model, max_tokens=max_tokens, temperature=temperature )
                    plan =  self.get_plan(output)

                    if self.safety:
                        input_text = input("Do you want to execute the plan? (y/Y) for YES: ")
                        if not input_text in ['y', 'Y']:
                            raise Exception("Plan execution aborted by user.")    
                    run_plan = self.run_plan(plan)
                    if plan[-1]['tool'].lower() == 'finish':
                        return plan
                    memory.append(plan)
                except Exception as e:
                    e = c.detailed_error(e)
                    print(f"Error in trial {trial + 1}: {e}", color='red')
                    memory.append(e)
        return plan

    def preprocess(self, text, src='./', magic_prefix = f'@'):

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
                        print(f"Could not parse {value} as JSON, using string.", color='yellow')
                        continue
                    fns[-1]['params'][key] = value
                    query += str(step['fn'](**step['params']))
                else:
                    fn_detected = False

        return query

    def load_step(self, text):
        text = text.split(self.anchors['tool'][0])[1].split(self.anchors['tool'][1])[0]
        try:
            step = json.loads(text)
        except json.JSONDecodeError:
            step = self.tool('fix_json')(text)
        return step

    def get_plan(self, output:str) -> list:
        text = ''
        plan = []
        for ch in output:
            text += ch
            c.print(ch, end='')
            is_plan_step = self.anchors['tool'][0] in text and self.anchors['tool'][1] in text
            if is_plan_step:
                plan.append(self.load_step(text))
                text = ''
        
        c.print("Plan:", plan, color='yellow')
        return plan

    def run_plan(self, plan: list) -> list:
        for i,step in enumerate(plan):
            if step['tool'].lower()  in ['finish', 'review']:
                print("Finishing as per plan.", color='green')
                break
            else:
                result = self.tool(step['tool'])(**step['params'])
        return plan

    def content(self, path: str = './', query=None, max_size=100000, timeout=20) -> List[str]:
        """
        Find files in a directory matching a specific pattern.
        """
        result = self.tool('select_files')(path=path, query=query, trials=4)
        content = str(result)
        size = len(content)
        c.print(f"path={path} max_size={max_size} size={size}", color='cyan')
        if size > max_size:
            summarize  = self.tool('sumfile')
            new_results = {}
            print(f"Content size {size} exceeds max_size {max_size}, summarizing...", color='red')
            futures = [c.submit(summarize, {'content': v, "query": query}, timeout=timeout) for k, v in result.items()]
            return c.wait(futures, timeout=timeout)
        else:
            result = content
        c.print(f"Content found: {len(result)} items", color='green')
        return result

    tool_prefix = 'dev.tool'

    tools_prefix = f"{__file__.split('/')[-2]}/tool"

    def tools(self, search=None, ignore_terms=['docker_image', 'content', 'select/files'], tools_prefix='dev.tool.', include_terms=[], update=False) -> List[str]:
        tools = c.mods(tools_prefix, folder=0)
        def filter_tool(tool: str) -> bool:
            global search
            if any(ignore in tool for ignore in ignore_terms):
                return False
            if any(term in tool for term in include_terms):
                return True
            return True
        result = []
        for tool in tools:
            tool = tool.replace(tools_prefix,  '')
            if filter_tool(tool):
                if search: 
                    if search not in tool:
                        continue

                result.append(tool)
        return result

        
    def toolbelt(self, verbose=True) -> Dict[str, str]:
        """
        Map each tool to its schema.
        
        Returns:
            Dict[str, str]: Dictionary mapping tool names to their schemas.
        """
        toolbelt = {}
        for t in self.tools():
            try:
                toolbelt[t.replace('.', '/')] = self.schema(t)
            except Exception as e:
                c.print(f"Error getting schema for tool {t}: {e}", color='red', verbose=verbose)
                continue
        return toolbelt
    
    def schema(self, tool: str, fn='forward') -> Dict[str, str]:
        """
        Get the schema for a specific tool.
        """
        return  c.schema(self.tool_prefix + '.' +tool.replace('/', '.'))[fn]

    def tool(self, tool_name: str='cmd', prefix='dev.tool', *args, **kwargs) -> Any:
        """
        Execute a specific tool by name with provided arguments.
        """
        tool_name = tool_name.replace('/', '.')
        return c.mod(prefix + '.' + tool_name)(*args, **kwargs).forward


    def add_docker_file(self, src='./'): 
        return self.forward( 'add a docker file given the following and do it in one shot', src=src, steps=1)


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