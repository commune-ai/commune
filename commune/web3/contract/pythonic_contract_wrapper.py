from functools import partial
import web3 
import streamlit as st
import gradio

class PythonicContractWrapper:
    def __init__(self, contract, account=None):

        for k,v in contract.__dict__.items():
            setattr(self, k, v)
        self.set_account(account)
        self.parse()
    
    def set_account(self, account=None):
        if account != None:
            self.account = account
            self.web3 = self.account.web3


    def parse(self):
        for fn_dict in self.functions._functions:
            fn_name = fn_dict['name']
            
            if fn_dict['stateMutability'] == 'view':
                def wrapped_fn(self,fn_name, *args, tx={}, **kwargs):
                    fn  = getattr(self.functions, fn_name)
                    return fn(*args, **kwargs).call()
            elif fn_dict['stateMutability'] in ['payable', 'nonpayable']:
                def wrapped_fn(self,fn_name, *args,tx={}, **kwargs):
                    value = tx.pop('value', 0)
                    fn  = getattr(self.functions, fn_name)
                    return self.account.send_contract_tx(fn(*args, **kwargs), value=value)

            else:
                raise NotImplementedError(fn_name)
            
            wrapped_fn_ = partial(wrapped_fn, self, fn_name)
            setattr(self,fn_name, wrapped_fn_)


    @property
    def function_schema(self):
        function_schema = {}
        for fn_abi in self.functions.abi:
            if fn_abi['type'] == 'constructor':
                name = 'constructor'
            elif fn_abi['type'] == 'function':
                name = fn_abi.pop('name')
            else:
                continue

            function_schema[name] = fn_abi
            
            for m in ['inputs', 'outputs']:
                if m in function_schema[name]:
                    function_schema[name][m] =  [{k:i[k] for k in ['type', 'name']} for i in function_schema[name][m]]

        return function_schema

    function_abi = function_schema

    @property
    def function_names(self):
        return list(self.function_schema.keys())


    def parser(self,  type=None, label=None):
        assert not type == None, "there should be a type to infer a gradio component"
        return {
            'string' : gradio.Textbox(label=label, lines=3, placeholder=f"Enter {label} here..."),
            'address': gradio.Textbox(label=label, lines=1, placeholder="0x0000000000000000000000000000000000000000"),
            'bytes'  : gradio.Textbox(label=label, lines=1, placeholder="bytes"),
            'uint256': gradio.Number(label=label, precision=None),
            'uint8'  : gradio.Number(label=label, precision=int),
            'bool'   : gradio.Checkbox(label=label)
        }[type]

    def package(self, inputs=[], outputs=[]):
        
        return ([self.parser(input['type'], "input" if input['name'] == "" else input['name']) for input in inputs], gradio.JSON(label="output") if outputs.__len__() > 0 else outputs)




    def gradio(self):
        import gradio

        
        fn, names = [], []
        abi = self.functions.abi
        for fn_name, fn_obj in self.function_schema.items():
            if fn_obj['type'] != 'function':
                continue 

            try:
                inp, out = self.package(fn_obj["inputs"], fn_obj["outputs"] if "outputs" in fn_obj else [])
            except KeyError:
                continue
            names.append(fn_name )
            fn.append(gradio.Interface(fn=getattr(self,fn_name), inputs=inp, outputs=out,))

        interface = gradio.TabbedInterface(fn, names)

        return interface


            
