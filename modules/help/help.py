import commune as c
class Help:
    def forward(self, *args, module=None):
        text = self.args2text(args)
        code_map = c.code_map(module)

        prompt = f'''
        and the code map: {code_map}
        help the user with the question: {text}
        '''
    def args2text(self, args):
        return ' '.join(map(str, args))
    

