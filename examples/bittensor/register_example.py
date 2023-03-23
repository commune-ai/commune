
import commune

if __name__ == "__main__":
    
    api = commune.api()
    api.launch('block.bittensor', fn='register_loop', mode='pm2', name='register_loop')
    # print(api.pm2_list())
