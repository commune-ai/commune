from signal import signal, SIGKILL
import gradio as gr
from inspect import getfile
import socket
import requests
import os 
class Dock:

    def __init__(self) -> None:
            self.port_map = dict()
            for p in range(7860, 7880):
                if not self.port_is_connected(p):
                    self.port_map[p] = True
                else:
                    self.port_map[p] = False

            

    def port_is_connected(self, port : int) -> bool:
        """
            @params: 
                - port : int
            @return:
                - boolean
            check if the port is open with in our localhost
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
            return s.connect_ex(("localhost", port)) == 0 


    def determinePort(self) -> any:
        """
            Take the port_map that was instantiate
            in the __init__ and loop though all ports
            and check if it the port is available  
        """
        for port, available in self.port_map.items():
            if available == True:
                if self.port_is_connected(port): # check if port is in use if so then go to the next one
                    continue
                return port
        
        raise Exception(f'❌ 🔌 {bcolor.BOLD}{bcolor.UNDERLINE}{bcolor.FAIL}All visable ports are used up...Try close some ports {bcolor.ENDC}')


        


DOCKER_LOCAL_HOST = '0.0.0.0' 
DOCKER_PORT = Dock() # Determine the best possible port 

# // =========================
# // = Decorator             =
# // =========================
def register(inputs, outputs, examples=None, **kwargs):
    """
        Decorator that is appended to a function either within a class or not
        and output either an interface or inputs and outputs for later processing
        to launch either to Gradio-Flow or just Gradio
    """
    def register_gradio(func):
        def decorator(*args, **wargs): 
            
            # if the output is a list then there should be equal or more then 1                   
            if type(outputs) is list:
                assert len(outputs) >= 1, f"❌ {bcolor.BOLD}{bcolor.FAIL}You have no bitches, and you have no outputs 🤨... {str(type(outputs))} {bcolor.ENDC}"
            
            fn_name = func.__name__ # name of the function

            # if there exist the self within the arguments and thats the first argument then this must be a class function
            if 'self' in func.__code__.co_varnames and func.__code__.co_varnames[0] == 'self' and fn_name in dir(args[0]):     
                """
                given the decorator is on a class then
                initialize a registered_gradio_functons
                if not already initialize.
                """
               
                # if the inputs is a list then inputs list should equal the arugments list
                if type(inputs) is list:
                    assert len(inputs) == func.__code__.co_argcount - 1, f"❌ {bcolor.BOLD}{bcolor.FAIL}inputs should have the same length as arguments{bcolor.ENDC}"

                try:
                    self = args[0]
                    self.registered_gradio_functons
                except AttributeError:
                    self.registered_gradio_functons = dict() # if registered_gradio_functons does not exist then create it 
                
                # if the function name is not within the registered_gradio_functons then register it within the registered_gradio_functons 
                if not fn_name in self.registered_gradio_functons:
                    self.registered_gradio_functons[fn_name] = dict(inputs=inputs,
                                                                    outputs=outputs, 
                                                                    examples=examples,
                                                                    cache_examples=kwargs['cache_examples'] if "cache_examples" in kwargs else None,
                                                                    examples_per_page=kwargs['examples_per_page'] if "examples_per_page" in kwargs else 10,
                                                                    interpretation=kwargs['interpretation'] if "interpretation" in kwargs else None,
                                                                    num_shap=kwargs['num_shap'] if "num_shap" in kwargs else 2.0,
                                                                    title=kwargs['title'] if "title" in kwargs else None,
                                                                    article=kwargs['article'] if "article" in kwargs else None,
                                                                    thumbnail=kwargs['thumbnail'] if "thumbnail" in kwargs else None,
                                                                    css=kwargs['css'] if "css" in kwargs else None,
                                                                    live=kwargs['live'] if "live" in kwargs else False,
                                                                    allow_flagging=kwargs['allow_flagging'] if "allow_flagging" in kwargs else None,
                                                                    theme=kwargs['theme'] if "theme" in kwargs else 'default', )

                # if the argument are within the function when it's called then give me the output of the function
                # giving the user the ability to use the function if necessary 
                if len(args[1:]) == (func.__code__.co_argcount - 1):
                    return func(*args, **wargs) 
                
                # return nothing if the arguments are not within it cause if the arguments do not exist it will give a error
                return None
            else :
                """
                the function is not a class function
                """

                # if the inputs is a list then inputs list should equal the arugments list
                if type(inputs) is list:    
                    assert len(inputs) == func.__code__.co_argcount, f"❌ {bcolor.BOLD}{bcolor.FAIL}inputs should have the same length as arguments{bcolor.ENDC}"

                # if the arguments within the functions are inputed then just return the output
                if len(args) == (func.__code__.co_argcount):
                    return func(*args, **wargs)

                # if there is nothing in the arugumrnt then return the gradio interface
                return gr.Interface(fn=func,
                                    inputs=inputs,
                                    outputs=outputs,
                                    examples=examples,
                                    cache_examples=kwargs['cache_examples'] if "cache_examples" in kwargs else None,
                                    examples_per_page=kwargs['examples_per_page'] if "examples_per_page" in kwargs else 10,
                                    interpretation=kwargs['interpretation'] if "interpretation" in kwargs else None,
                                    num_shap=kwargs['num_shap'] if "num_shap" in kwargs else 2.0,
                                    title=kwargs['title'] if "title" in kwargs else None,
                                    article=kwargs['article'] if "article" in kwargs else None,
                                    thumbnail=kwargs['thumbnail'] if "thumbnail" in kwargs else None,
                                    css=kwargs['css'] if "css" in kwargs else None,
                                    live=kwargs['live'] if "live" in kwargs else False,
                                    allow_flagging=kwargs['allow_flagging'] if "allow_flagging" in kwargs else None,
                                    theme='default', 
                                    )
        decorator.__decorator__ = "__gradio__" # siginture to tell any function that need to know that function is a registed gradio application
        return decorator
    return register_gradio

def GradioModule(cls):
    class Decorator:

        def __init__(self, *args, **kwargs) -> None:
            self.__cls__ = cls(*args, **kwargs)
            self.__get_funcs_attr()
            self.interface = self.__compile()
        
        def get_funcs_names(self):
            """
                Get all name for each function
            """
            assert self.get_registered_map() != None, "this is not possible..."
            return [ name for name in self.get_registered_map().keys()]

        def get_registered_map(self):
            """
                Get all registered functions
            """
            assert self.__cls__.registered_gradio_functons != None, "what happen!!!!"
            return self.__cls__.registered_gradio_functons
        
        def __get_funcs_attr(self):
            """
                Get all the function that are registered
            """
            for func in dir(self.__cls__):
                fn = getattr(self.__cls__, func, None)
                
                if callable(fn) and not func.startswith("__") and  "__decorator__" in dir(fn) and fn.__decorator__ == "__gradio__":
                    fn()

        def __compile(self):
            """
            Initialize all the function 
            within the class that are registeed
            """
            demos, names = [], []
            for func, param in self.get_registered_map().items(): # loop though the registered function and append it to the TabularInterface           
                names.append(func) 
                try:
                    demos.append(gr.Interface(fn=getattr(self.__cls__, func, None), **param))
                except Exception as e :
                    raise e

            print(f"{bcolor.OKBLUE}COMPLETED: {bcolor.ENDC}All functions are mapped, and ready to launch 🚀",
                 "\n===========================================================\n")
            return gr.TabbedInterface(demos, names)
            
        def launch(self, **kwargs):
            """
                @params:
                    **kwargs
                Take the tabular interface and send it to the api if
                'listen' is within the kwargs and launch the gradio interface
                then when the gradio stops then remove it from the api
            """
            port= kwargs["port"] if "port" in kwargs else DOCKER_PORT.determinePort() 
            if 'listen' in kwargs:
                try:
                    requests.post(f"http://{DOCKER_LOCAL_HOST}:{ kwargs[ 'listen' ] }/api/append/port", json={"port" : port, "host" : f'http://localhost:{port}', "file" : getfile(self.__cls__.__class__), "name" : self.__cls__.__class__.__name__, "kwargs" : kwargs})
                except Exception:
                    print(f"**{bcolor.BOLD}{bcolor.FAIL}CONNECTION ERROR{bcolor.ENDC}** 🐛The listening api is either not up or you choose the wrong port.🐛")
                    return

            self.interface.launch(server_port=port,
                                  server_name=f"{DOCKER_LOCAL_HOST}",
                                  inline= kwargs['inline'] if "inline" in kwargs else None,
                                  share=kwargs['share'] if "share" in kwargs else None,
                                  debug=kwargs['debug'] if "debug" in kwargs else False,
                                  enable_queue=kwargs['enable_queue'] if "enable_queue" in kwargs else None,
                                  max_threads=kwargs['max_threads'] if "max_threads" in kwargs else None,
                                  auth=kwargs['auth'] if "auth" in kwargs else None,
                                  auth_message=kwargs['auth_message'] if "auth_message" in kwargs else None,
                                  prevent_thread_lock=kwargs['prevent_thread_lock'] if "prevent_thread_lock" in kwargs else False,
                                  show_error=kwargs['show_error'] if "show_error" in kwargs else True,
                                  show_tips=kwargs['show_tips'] if "show_tips" in kwargs else False,
                                  height=kwargs['height'] if "height" in kwargs else 500,
                                  width=kwargs['width'] if "width" in kwargs else 900,
                                  encrypt=kwargs['encrypt'] if "encrypt" in kwargs else False,
                                  favicon_path=kwargs['favicon_path'] if "favicon_path" in kwargs else None,
                                  ssl_keyfile=kwargs['ssl_keyfile'] if "ssl_keyfile" in kwargs else None,
                                  ssl_certfile=kwargs['ssl_certfile'] if "ssl_certfile" in kwargs else None,
                                  ssl_keyfile_password=kwargs['ssl_keyfile_password'] if "ssl_keyfile_password" in kwargs else None,
                                  quiet=kwargs['quiet'] if "quiet" in kwargs else False) 
            if 'listen' in kwargs:
                try:
                    requests.post(f"http://{DOCKER_LOCAL_HOST}:{ kwargs[ 'listen' ] }/api/remove/port", json={"port" : port, "host" : f'http://localhost:{port}', "file" : getfile(self.__cls__.__class__), "name" : self.__cls__.__class__.__name__, "kwargs" : kwargs})
                except Exception:
                    print(f"**{bcolor.BOLD}{bcolor.FAIL}CONNECTION ERROR{bcolor.ENDC}** 🐛The api either lost connection or was turned off...🐛")
            return

    return Decorator

# console colour changer 
class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

