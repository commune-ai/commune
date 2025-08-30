import commune as c
class Help:
    def forward(self, *args, module=None):
        code_map = c.context(module)

        prompt = f'''
        and the code map: {code_map}
        help the user with the question: {text}
        '''
        return c.mod('openrouter')().forward(prompt)
    def args2text(self, args):
        return ' '.join(map(str, args))
    

