



class Logger:
    
    
    
    word2color_map = {
        'error': 'red'
    }
    
    
    
    color2code_map = {
        'green': '92',
        'yellow': '93',
        'purple': '95',
        # continue
        'red': '91',
        'white': '97',
        'blue': '94',
        'cyan': '96',
        'grey': '90',
        'black': '90',
        'lightgrey': '97',
        
    }
    
    
    @classmethod
    def print(cls, text:str, color:str = 'white', return_text:bool = False):
        color_code = cls.color2code_map[color]
        print_text = f'\033[{color_code}m {text}\033[00m'
        if return_text:
            return print_text
        else:
            print(print_text)
