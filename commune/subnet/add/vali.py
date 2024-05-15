import commune as c

class Subnet(c.m('vali')):
    def __init__(self, network='local', search='add', **kwargs):
        self.init_vali(locals())

    def score_module(self, module):
        a = c.random_int()
        b = c.random_int()
        output = module.forward(a, b)
        if output == a + b:
            return 1
        else:
            return 0
        
    def test(self, n=3, sleep_time=3):
        test_miners = ['subnet.miner.add::test' for i in range(n)]
        for miner in test_miners:
            c.serve(miner)
        test_vali = 'subnet.vali.add::test'
        for miner in test_miners:
            c.serve(test_vali, kwargs={'network': 'local'})
        
        c.print('Sleeping for 3 seconds')
        c.sleep(sleep_time)

        leaderboard = c.call(test_vali+'/leaderboard')

        c.print(leaderboard)
        assert len(leaderboard) == n
        


        c.serve('subnet.miner::test')