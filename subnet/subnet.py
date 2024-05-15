import commune as c

class Subnet(c.Module):
    def score_module(self, module):
        a = c.random_int()
        b = c.random_int()
        output = module.forward(a, b)

        if output == a + b:
            return 1
        else:
            return 0
        

    def test(self, n=3, sleep_time=3):
        test_vali = 'subnet.vali::test'
        test_miners = [f'subnet.miner::test_{i}' for i in range(n)]
        for miner in test_miners:
            c.serve(miner)
        c.serve(test_vali, kwargs={'network': 'local', 'search': 'miner::test_'})
        
        c.print('Sleeping for 3 seconds')
        c.sleep(sleep_time)

        leaderboard = c.call(test_vali+'/leaderboard')



        c.print(leaderboard)
        
        c.serve('subnet.miner::test')

    