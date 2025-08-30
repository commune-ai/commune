import commune as c
import json
import os
import inspect
from typing import List, Dict, Union, Optional, Any

print = c.print

class Tool:

    def forward(self,
                module_path: str,
                fn_name: str,
                fn_code: str,
                class_name: str = 'Tool',
                **kwargs) -> str:
        """
        Add a function to a module class
        
        Args:
            module_path: Path to the module file (e.g. 'dev.tool.rm_content')
            fn_name: Name of the function to add
            fn_code: The function code to add
            class_name: Name of the class to add the function to (default: 'Tool')
        """
        # Convert module path to file path
        if '.' in module_path and not module_path.endswith('.py'):
            file_path = c.filepath(module_path)
        else:
            file_path = os.path.abspath(module_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Module file not found: {file_path}")
        
        # Read the file content
        content = c.text(file_path)
        
        # Find the class definition
        class_pattern = f"class {class_name}"
        if class_pattern not in content:
            raise ValueError(f"Class '{class_name}' not found in {file_path}")
        
        # Find the last method in the class
        lines = content.split('\n')
        class_start_idx = None
        last_method_end_idx = None
        current_indent = None
        in_class = False
        
        for i, line in enumerate(lines):
            if class_pattern in line and ':' in line:
                class_start_idx = i
                in_class = True
                continue
            
            if in_class:
                # Check if we've left the class (dedented back)
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    break
                
                # Look for method definitions
                if line.strip().startswith('def '):
                    current_indent = len(line) - len(line.lstrip())
                    # Find the end of this method
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j]
                        if next_line.strip():  # Non-empty line
                            next_indent = len(next_line) - len(next_line.lstrip())
                            # If we're back at the same indent level or less, method ended
                            if next_indent <= current_indent and next_line.strip().startswith('def '):
                                last_method_end_idx = j - 1
                                break
                            elif next_indent < current_indent:
                                last_method_end_idx = j - 1
                                break
                    else:
                        # Reached end of file
                        last_method_end_idx = len(lines) - 1
        
        if last_method_end_idx is None:
            raise ValueError(f"Could not find methods in class '{class_name}'")
        
        # Prepare the new function code with proper indentation
        class_indent = '    '  # Standard 4-space indent for class methods
        indented_fn_lines = []
        fn_lines = fn_code.strip().split('\n')
        
        for line in fn_lines:
            if line.strip():  # Non-empty line
                if line.strip().startswith('def '):
                    indented_fn_lines.append(class_indent + line.strip())
                else:
                    # Maintain relative indentation
                    indented_fn_lines.append(class_indent + line)
            else:
                indented_fn_lines.append('')  # Empty line
        
        # Insert the new function after the last method
        new_lines = lines[:last_method_end_idx + 1]
        new_lines.append('')  # Add blank line before new method
        new_lines.extend(indented_fn_lines)
        new_lines.extend(lines[last_method_end_idx + 1:])
        
        # Write the updated content back to the file
        new_content = '\n'.join(new_lines)
        c.put_text(file_path, new_content)
        
        c.print(f"Successfully added function '{fn_name}' to {file_path}", color='green')
        return new_content

    def fn_schema(self, fn: str = '__init__', code=True, **kwargs) -> dict:
        '''
        Get function schema of function in self
        '''
        schema = {}
        fn_obj = self.fn(fn)
        if not callable(fn_obj):
            return {'fn_type': 'property', 'type': type(fn_obj).__name__}
        fn_signature = inspect.signature(fn_obj)
        schema['input'] = {}
        for k, v in dict(fn_signature._parameters).items():
            schema['input'][k] = {
                'value': "_empty" if v.default == inspect._empty else v.default,
                'type': '_empty' if v.default == inspect._empty else str(type(v.default)).split("'")[1]
            }
        schema['output'] = {
            'value': None,
            'type': str(fn_obj.__annotations__.get('return', None) if hasattr(fn_obj, '__annotations__') else None)
        }
        schema['docs'] = fn_obj.__doc__
        schema['cost'] = 1 if not hasattr(fn_obj, '__cost__') else fn_obj.__cost__
        schema['name'] = fn_obj.__name__
        schema.update(self.source(fn_obj, code=code))
        return schema

    def schema(self, obj=None, verbose=False, **kwargs) -> dict:
        '''
        Get function schema of function in self
        '''
        schema = {}
        obj = obj or 'module'
        if callable(obj):
            return self.fn_schema(obj, **kwargs)
        elif isinstance(obj, str):
            if '/' in obj:
                return self.fn_schema(obj, **kwargs)
            elif self.module_exists(obj):
                obj = self.mod(obj)
            elif hasattr(self, obj):
                obj = getattr(self, obj)
                schema = self.fn_schema(obj, **kwargs)
            else:
                raise Exception(f'{obj} not found')
        elif hasattr(obj, '__class__'):
            obj = obj.__class__
        for fn in self.fns(obj):
            try:
                schema[fn] = self.fn_schema(getattr(obj, fn), **kwargs)
            except Exception as e:
                self.print(f'Error {e} {fn}', color='red', verbose=verbose)
        return schema

    def addyhelloworld(self, a: int, b: int) -> str:
        """
        Says hello world and adds two numbers together.

        Args:
            a: First number to add
            b: Second number to add

        Returns:
            A string with hello world message and the sum
        """
        result = a + b
        return f"Hello World! The sum of {a} and {b} is {result}"