
import commune

if __name__ == "__main__":
    
    api = commune.api()
    block = api.launch('block.bittensor', fn='register_loop',mode='pm2', kwargs=dict(subtensor='nobunaga'), name='register_loop_test')
    # print(api.pm2_list())
