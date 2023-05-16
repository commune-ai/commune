import commune as c

# bt = c.connect('bittensor')
# c.print(bt.ss58())

w = [
    'miner::ensemble.0::local::3',
    'miner::ensemble.10::local::3',
    'miner::ensemble.12::local::3',
    'miner::ensemble.13::local::3',
    'miner::ensemble.14::local::3',
    'miner::ensemble.15::local::3',
    'miner::ensemble.1::local::3',
    'miner::ensemble.2::local::3'
]

for _ in w:
    c.kill(_)