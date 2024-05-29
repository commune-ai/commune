import commune as c
import pandas as pd

class Subnet(c.Module):
    def score_module(self, module):
        a = c.random_int()
        b = c.random_int()
        output = module.forward(a, b)

        if output == a + b:
            return 1
        else:
            return 0
        

    @classmethod
    def test(cls, n=3, sleep_time=4):
        test_miners = [f'subnet.miner::test_{i}' for i in range(n)]
        for miner in test_miners:
            c.print(c.serve(miner))

        test_vali = 'subnet.vali::test'
        c.serve(test_vali, kwargs={'network': 'local', 'search': 'miner::test_'})
        
        c.print(f'Sleeping for {sleep_time} seconds')
        c.sleep(sleep_time)

        leaderboard = c.call(test_vali+'/leaderboard')
        assert isinstance(leaderboard, pd.DataFrame), leaderboard
        assert len(leaderboard) == n, leaderboard
        c.print(c.call(test_vali+'/refresh_leaderboard'))

        c.print(leaderboard)
        
        c.serve('subnet.miner::test')
        for miner in test_miners + [test_vali]:
            c.print(c.kill(miner))
        return {'success': True, 'msg': 'subnet test passed'}