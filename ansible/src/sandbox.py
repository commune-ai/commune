import commune

if __name__ == "__main__":
    
    # api = commune.api(refresh=False)
    # api.launch('block.bittensor', fn='register_loop', mode='pm2', name='register_loop')
    import ray
    
    print(ray.get(commune.ray_actor('module', virtual=False).pm2_status.remote()))
    # print(api.pm2_list())
    # print(api.pm2_list())
