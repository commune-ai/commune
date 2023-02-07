import commune
class DemoClass:
    fam = True

    def bro(self, data:str):
        '''
        brooo ergergre
        '''
        return data
    
def test_create_actor():
    
    # commune.module(DemoClass)().serve()
    commune.connect('DemoClass').function_schema_map(fam='djjfjfj')

# print(commune.module('ReactAgentModule').run('What is the capital of the United States?'))
# agent.serve(wait_for_termination=True)
# print(commune.connect('ModuleF'))
test_create_actor()