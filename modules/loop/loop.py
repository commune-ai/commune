import commune as c

class Loop(c.Module):

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    

    def loops(self, module2timeout= {'module': 10, 'subspace': 10}):
        t1 = c.timestamp()
        while  True:
            t2 = c.timestamp()
            for module, timeout in module2timeout.items():
                if t2 - t1 > timeout:
                    c.update(module=module)

    def app(self):
        import streamlit as st

        loops = self.loops()
        modules = ['module', 'subspace', 'remote']
        option = st.selectbox('Select a Loop', modules, modules.index('module'), key=f'loop.option')
        if option == 'module':
            loop_name = st.selectbox('Select a Loop', loops, loops.index('loop'), key=f'loop.loop_name')
            loop = c.loop(loop_name)
            loop.dashboard()
        elif option == 'subspace':
            loop_name = st.selectbox('Select a Loop', loops, loops.index('loop'), key=f'loop.loop_name')
            loop = c.loop(loop_name)
            loop.dashboard()
        elif option == 'remote':
            loop_name = st.selectbox('Select a Loop', loops, loops.index('loop'), key=f'loop.loop_name')
            loop = c.loop(loop_name)
            loop.dashboard()
        else:
            raise Exception(f'Invalid option {option}')


Loop.run(__name__)
    