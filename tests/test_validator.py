
import commune as c
import pandas as pd

def test_net( 
        n=2, 
            sleep_time=8, 
            timeout = 20,
            tag = 'vali_test_net',
            miner='module', 
            vali='vali', 
            storage_path = '/tmp/commune/vali_test',
            network='local'):
    

    
    test_miners = [f'{miner}::{tag}_{i}' for i in range(n)]
    test_vali = f'{vali}::{tag}'
    modules = test_miners + [test_vali]
    for m in modules:
        c.kill(m) 
    for m in modules:
        c.print(c.serve(m, kwargs={'network': network, 
                                    'storage_path': storage_path,
                                    'search': test_miners[0][:-1]}))
    t0 = c.time()
    while not c.server_exists(test_vali):
        time_elapsed = c.time() - t0
        if time_elapsed > timeout:
            return {'success': False, 'msg': 'subnet test failed'}
        c.sleep(1)
        c.print(f'Waiting for {test_vali} to get the Leaderboard {time_elapsed}/{timeout} seconds')

    t0 = c.time()
    c.print(f'Sleeping for {sleep_time} seconds')
    c.print(c.call(test_vali+'/refresh_leaderboard'))
    leaderboard = None
    while c.time() - t0 < sleep_time:
        try:
            vali = c.connect(test_vali)
            leaderboard = c.call(test_vali+'/leaderboard')

            if len(leaderboard) >= n:
                break
            else:
                c.print(f'Waiting for leaderboard to be updated {len(leaderboard)} is n={n}')
            c.sleep(1)
        except Exception as e:
            print(e)

    leaderboard = c.call(test_vali+'/leaderboard', df=1)
    assert isinstance(leaderboard, pd.DataFrame), leaderboard
    assert len(leaderboard) >= n, leaderboard
    c.print(c.call(test_vali+'/refresh_leaderboard'))        
    for miner in test_miners + [test_vali]:
        c.print(c.kill(miner))
    return {'success': True, 'msg': 'subnet test passed'}
