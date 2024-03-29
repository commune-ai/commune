import commune as c

class Tool(c.Module):

    def tools(self, fn_name='call'):
        tools = []
        for module in c.tqdm(c.modules()):
            try:
                if hasattr(c.module(module),fn_name):
                    tools.append(module)
            except Exception as e:
                pass
        return tools

    def has_tool(self, module, fn_name='call'):
        try:
            return f'def {fn_name}(' in c.code(module)
        except:
            return False
        
     
    





    
