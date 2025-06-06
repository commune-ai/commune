
import commune as c
class Test:
    def test_basics(self ,  n=2, 
                tag = 'vali_test_net',  
                miner='module', 
                trials = 20,
                tempo = 1,
                update=True,
                path = '/tmp/commune/vali_test',
                network='local'
                ):
            Vali  = c.module('vali')
            test_miners = [f'{miner}::{tag}{i}' for i in range(n)]
            modules = test_miners
            search = tag
            assert len(modules) == n, f'Number of miners not equal to n {len(modules)} != {n}'
            for m in modules:
                c.serve(m)
            vali = Vali(network=network, search=search, path=path, update=update, tempo=tempo, run_loop=False)
            scoreboard = []
            while len(scoreboard) < n:
                c.sleep(1)
                scoreboard = vali.epoch(update=update)
                trials -= 1
                assert trials > 0, f'Trials exhausted {trials}'
            for miner in modules:
                c.print(c.kill(miner))
            assert c.server_exists(miner) == False, f'Miner still exists {miner}'
            return {'success': True, 'msg': 'subnet test passed'}