import commune as c

class Test:
    @staticmethod
    def test(   n=2, 
                tag = 'vali_test_net',  
                miner='module', 
                trials = 5,
                tempo = 4,
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
            namespace = c.namespace()
            for m in modules:
                assert m in namespace, f'Miner not in namespace {m}'
            vali = Vali(network=network, search=search, path=path, update=update, tempo=tempo, run_loop=False)
            scoreboard = []
            while len(scoreboard) < n:
                c.sleep(1)
                scoreboard = vali.epoch()
                trials -= 1
                assert trials > 0, f'Trials exhausted {trials}'
            for miner in modules:
                c.print(c.kill(miner))
            return {'success': True, 'msg': 'subnet test passed'}