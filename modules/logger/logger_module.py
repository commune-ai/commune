



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
    def log(cls, text:str, color:str = 'white'):
        color_code = cls.color2code_map[color]
        return f'\033[{color_code}m {text}\033[00m'