# edit.py
import commune as c
import time
import os
import json
from typing import Dict, Any, Union
from .tools import add_file, delete_file, add_between, delete_between
class Edit:
    # Anchors for parsing model output
    fn_anchor = 'FN_CALL'
    fn_param_anchor = 'FN_PARAM'
    start_fn_anchor = f'<START_{fn_anchor}>'
    end_fn_anchor = f'<END_{fn_anchor}>'
    endpoints = ["forward"]

    def __init__(self, 
                model = 'anthropic/claude-3.5-sonnet',
                tools=[add_file, delete_file, add_between, delete_between]):
        self.model = model
        self.agent = c.module('agent')(model=self.model)
        self.models = self.agent.models()
        self.tools = {f.__name__: f for f in tools}
        self.tool2schema = {f.__name__: c.fn_schema(f) for f in tools}

    def forward(self,
                text='edit the file',
                *extra_text,
                path='./commune',
                task=None,
                temperature=0.5,
                module=None,
                max_tokens=1000000,
                threshold=1000000,
                model=None,
                write=False,
                process_text=False,
                stream=True):
        
        model = model or self.model
        text = text + ' ' + ' '.join(list(map(str, extra_text)))
        if module:
            path = c.filepath(module)

        if path != None:
            context = c.text(path)
        else:
            context = 'No file path provided'
        # Construct function documentation for the model
        fn_docs = "\n".join([f"{name}: {schema}" for name, schema in self.tool2schema.items()])
        
        prompt = f"""
            ---GOAL---
            You are an expert code editor. You will suggest changes to files using the available tools:
            ---CONTEXT---
            {text}
            {context}
            ---TOOLS---
            {fn_docs}
            ---OUTPUTFORMAT---
            - Provide complete solutions
            - Call tools in a logical sequence
            - Each function call will be confirmed by the user
            - make sure to add before you delete
            MAKE SURE THE OUTPOUT JSON IS BETWEEN THE ANCHORS: 
            DO NOT MENTION THE ANCHORS WITHIN THE ANCHORS FOR PARSING
            MAKE SURE TO PASS THE NAME OF EACH PARAM, THE TYPE AND THE VALUE UNDERNEATH
            SEND THE KEYWORD ARGUMENTS IN THE ORDER THEY ARE DEFINED IN THE FUNCTION
            {self.start_fn_anchor}
            --FN_NAME--
            add_lines
            --{self.fn_param_anchor}/varname:str--
            fhey
            --{self.fn_param_anchor}/varname:int--
            10
            {self.end_fn_anchor}

        """


        
        output = self.agent.generate(prompt, 
                                   stream=stream, 
                                   model=model, 
                                   max_tokens=max_tokens, 
                                   temperature=temperature, 
                                   process_text=process_text)
        
        return self.process_output(output, write=write)

    def parse_operation(self, fn_call_text: str) -> tuple[str, dict]:
        """
        Parse function call text with improved error handling and typing.
        
        params:
            fn_call_text: Raw function call text to parse
            
        Returns:
            Tuple of (function_name, parameters)
        """
        lines = fn_call_text.split('\n')
        fn_name = None
        params = {}
        current_key = None
        current_type = None
        current_value = []

        try:
            for line in lines:
                if line.startswith('--FN_NAME--'):
                    fn_name = next(l for l in lines[lines.index(line)+1:] if l.strip()).strip()
                elif line.startswith(f'--{self.fn_param_anchor}/'):
                    if current_key and current_value:
                        params[current_key] = {'type': current_type, 'value': '\n'.join(current_value)}
                    # Parse key and type from format --FN_ARG🔑TYPE--
                    parts = line[len(f'--{self.fn_param_anchor}/'):-2].split(':')
                    current_key = parts[0]
                    current_type = parts[1] if len(parts) > 1 else 'str'
                    current_value = []
                elif line and current_key is not None:
                    current_value.append(line)

            if current_key and current_value:
                params[current_key] = {'type': current_type, 'value': '\n'.join(current_value)}

        except Exception as e:
            c.print(f"Error parsing function call: {e}", color='red')
            return None, {}
        params = {
                key: int(param['value']) if param['type'] == 'int' else param['value']
                for key, param in params.items()
            }

        return {'fn': fn_name, 'params': params}

    def execute_operation(self, op:Dict) -> Dict:
        """Execute a single operation with proper error handling."""
        fn_name = op['fn']
        params = op['params']
        try:
            if fn_name not in self.tools:
                raise ValueError(f"Unknown function: {fn_name}")
            tool = self.tools[fn_name]
            result = tool(**params)
            return {
                "operation": fn_name,
                "params": params,
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "operation": fn_name,
                "params": params,
                "status": "failed",
                "error": str(e)
            }

    def process_output(self, response: Union[str, iter], write: bool = False) -> Dict[str, Any]:
        """Process model output with improved error handling and typing."""
        color = c.random_color()
        content = ''
        ops = []
        results = []

        try:
            for token in response:
                content += str(token)
                c.print(token, end='', color=color)

                while self.start_fn_anchor in content and self.end_fn_anchor in content:
                    start_idx = content.find(self.start_fn_anchor)
                    end_idx = content.find(self.end_fn_anchor)
                    fn_call = content[start_idx + len(self.start_fn_anchor):end_idx]
                    op = self.parse_operation(fn_call)
                    ops.append(op)
                    c.print(op)
                    content = content[end_idx + len(self.end_fn_anchor):]

            if ops:
                c.print("\nProposed ops:", color='yellow')
                c.print(ops, color='green')
                
                if input("\nExecute all ops? (y/n): ").lower() == 'y':
                    for op in ops:
                        c.print(f"\nExecuting: {op}", color='blue')
                        if input("Proceed with this operation? (y/n): ").lower() == 'y':
                            result = self.execute_operation(op)
                            results.append(result)
                            failed = result['status'] == 'failed'
                            if failed:
                                c.print(f"Operation failed: {result['error']}", color='red')
                            else:
                                c.print(f"Operation {result['status']}", color='green')
                        else:
                            results.append({
                                "operation": fn_name,
                                "params": params,
                                "status": "skipped"
                            })
                else:
                    results = [{
                        "operation": op[0],
                        "params": op[1],
                        "status": "cancelled"
                    } for op in ops]

        except Exception as e:
            c.print(f"Error processing output: {e}", color='red')
            results.append({
                "status": "failed",
                "error": str(e)
            })

        return {
            'ops': ops,
            'results': results,
            'write': write
        }
