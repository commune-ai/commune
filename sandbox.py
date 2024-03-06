import argparse
import commune as c


s = c.module('subspace')()
modules = ['5GTYocYQoQLFPVNZNN7TVzddwb3v7GZCGs2jbPvg4ja18LTF', '5FCmNmPNrdEf4hUionkibgdPcT4M9MKoKcpkonceck4UxFK5', '5GvRUs7HSNQgwnS2qhSqRKt4ihkcCUoaUyGU68H9vnk6H4VJ']
s.unstake_many(modules=modules, amounts=100000)

