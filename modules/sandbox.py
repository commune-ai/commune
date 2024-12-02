import commune as c 

data = {
    'data': '{"data": "fam laoniu CDL mining", "time": 1732752000}',
    'crypto_type': 1,
    'signature': 'd25da45e666449f8797786e2d86ba4758c393985bf8af0dd2cc055c21a38cb30b46114b0514748e0ba41ef05e5c99cb193eb5a239f15dbc684656f1d0cc14280',
    'address': '5DX4ytqpzDQmEfD9mxq5Gs7FaNnEpCiJh6s1qJkJzahGx8LC'
}
print(c.verify(data))