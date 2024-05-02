import commune as c

batches = [
    {
        'ss58_address': '5EjmjSB4jEB5eLfos1GtqvRgoXpxUKFmCTRCMLk9vPGTdzuB',
        'name': 'storage::tang15',
        'address': 'http://24.83.27.62:50073'
    },
    {
        'ss58_address': '5H6WCKw9Nvtx49Dq3TYPHR2qGDQWPzy9wcBB6qYMeg9481Jn',
        'name': 'storage::tang14',
        'address': 'http://24.83.27.62:50086'
    },
    {
        'ss58_address': '5GYqAH1yJsCLoq8zwHLN336huDnK7RAFqSahLPihiGNg6kQr',
        'name': 'storage::tang1',
        'address': 'http://24.83.27.62:50146'
    },
    {
        'ss58_address': '5F1MYAiFSguxgAuXPHrotbPv9DZqE6VRohwBJBYZUwvS6f47',
        'name': 'storage::tang16',
        'address': 'http://24.83.27.62:50088'
    },
    {
        'ss58_address': '5DPB1G99txdPBK8MZRKz11Pr5vFzgxEcezYSnmw2btsahJ1Y',
        'name': 'storage::tang10',
        'address': 'http://24.83.27.62:50135'
    },
    {
        'ss58_address': '5FxsyCxPmgArMCWQr9JQPggbDDYwoZ4iW1nabjEzoJhMGL41',
        'name': 'storage::tang0',
        'address': 'http://24.83.27.62:50094'
    },
    {
        'ss58_address': '5FZU5hJUVcHnERp3U4dkBdH2Rq3SCet5ztZUKVUZbn2r2eRQ',
        'name': 'storage::tang13',
        'address': 'http://24.83.27.62:50050'
    },
    {
        'ss58_address': '5FjrgnrfCV974z59uHRBHiLJYJdiKui8Mj3rNdwsxW8LkmaD',
        'name': 'storage.vali::tang',
        'address': 'http://24.83.27.62:50075'
    },
    {
        'ss58_address': '5FTimeCSYfgiYP7NfKzNCRYwTyoFMvfqrLPRv9LtcZEiix8j',
        'name': 'storage::tang12',
        'address': 'http://24.83.27.62:50144'
    },
    {
        'ss58_address': '5CGRkaz5RLT7nCPszTkgbneK6Zj53eXdW1FsdiSzz45jBLxg',
        'name': 'storage::tang11',
        'address': 'http://24.83.27.62:50125'
    },
    {
        'ss58_address': '5ECQZzYyW6Rfod6szbU3dE5VXwT3XBXHDvFdipxfsoLkKJLC',
        'name': 'storage::tang17',
        'address': 'http://24.83.27.62:50109'
    },
    {
        'ss58_address': '5EUPQiJtDdkJwURzmjFhY95JhsZB2vdhavsi2vDc2zQc82vi',
        'name': 'storage::tang18',
        'address': 'http://24.83.27.62:50119'
    },
    {
        'ss58_address': '5Ef2VuyqYf9Vmir3kT1VsPJgo8aTXjVRrY17TrTmvqA6gPh6',
        'name': 'storage::tang19',
        'address': 'http://24.83.27.62:50116'
    },
    {
        'ss58_address': '5DX1EaedRwHJhTgfjVjQhw1mZLVCwizopKEgDESeMmbZzD3S',
        'name': 'storage::tang25',
        'address': 'http://24.83.27.62:50123'
    },
    {
        'ss58_address': '5ExjSEHZbjemchT68CWWpn8XgA67muMS7WL2iX3bRWZf6EQj',
        'name': 'storage::tang22',
        'address': 'http://24.83.27.62:50141'
    },
    {
        'ss58_address': '5Da4sBpCHRf39AeZMg26T5CQAijQnU1WXR81CgM2crSqXUzr',
        'name': 'storage::tang20',
        'address': 'http://24.83.27.62:50102'
    },
    {
        'ss58_address': '5GgvJYE1CbtuLfwoSb3W1JjvX7XWtJM8YZqrudnLEGqY1WoT',
        'name': 'storage::tang21',
        'address': 'http://24.83.27.62:50093'
    },
    {
        'ss58_address': '5Hp6ccfnLa77YCydDw9xvsyPcXYbqHo4tzLMRXpxMSNpt7Ju',
        'name': 'storage::tang24',
        'address': 'http://24.83.27.62:50057'
    },
    {
        'ss58_address': '5C8NPhW9tn6uVQCqb6yxjgwfjnbZdM2YQkD9FWkqXyoHtRu7',
        'name': 'storage::tang23',
        'address': 'http://24.83.27.62:50095'
    },
    {
        'ss58_address': '5CkKL9j2S57HwYmLTmDuyrxTwqPyx98k88oMQzESpCjSF1MX',
        'name': 'storage::tang26',
        'address': 'http://24.83.27.62:50101'
    },
    {
        'ss58_address': '5EvfcgVmXfX2fg5esyeDyz3ban3BUKsSDd2TRKyC7386bF8C',
        'name': 'storage::tang2',
        'address': 'http://24.83.27.62:50090'
    },
    {
        'ss58_address': '5DeaZ9QtWz7KKWUriMLhZVocvTru3JcKz78wBX7F4JPAEem8',
        'name': 'storage::tang27',
        'address': 'http://24.83.27.62:50139'
    },
    {
        'ss58_address': '5ED61Hj5d14SwDvp4kXvfiHNknoxt9N9LpWEs1hyG65WwQwe',
        'name': 'storage::tang28',
        'address': 'http://24.83.27.62:50145'
    },
    {
        'ss58_address': '5EqYeVi18AUZmjcSprQqDbSBayT4ctBtzyW8LtMeMjoVfJdt',
        'name': 'storage::tang4',
        'address': 'http://24.83.27.62:50063'
    },
    {
        'ss58_address': '5F2HaZMSYH24p8k35gTnvTtksYZmF6cVaXHhsjiK7fz6HWdG',
        'name': 'storage::tang3',
        'address': 'http://24.83.27.62:50076'
    },
    {
        'ss58_address': '5CwCWvMsYy3bDuoDwH5NuKmiyw3Dx8jz3TVZH3pdrQZFgDfp',
        'name': 'storage::tang5',
        'address': 'http://24.83.27.62:50074'
    },
    {
        'ss58_address': '5F4zZKKQXrmqWPbGkTzbzbquRdJP8PgjEau6sPNTZVWtoESp',
        'name': 'storage::tang7',
        'address': 'http://24.83.27.62:50085'
    },
    {
        'ss58_address': '5Hmw2bPoaRjKapWcSfsEiCoiG7bRj9p4KWjTcQ4AJD5BXZMz',
        'name': 'storage::tang8',
        'address': 'http://24.83.27.62:50111'
    },
    {
        'ss58_address': '5FvVW7Ey9rhv2vjV1yFkRCCN8t5HscqS7shmUggmUncyWT5H',
        'name': 'storage::tang29',
        'address': 'http://24.83.27.62:50121'
    },
    {
        'ss58_address': '5GeeGRiLktWr1W8KLNfmN69JVgYFJPkgK2nPXy8eDeFLvM7L',
        'name': 'storage::tang9',
        'address': 'http://24.83.27.62:50054'
    },
    {
        'ss58_address': '5H8rJ1cn87Qxfz9Jx6rHY6UKp7ypaaJjgcKXLyduBshFW4fr',
        'name': 'storage::tang6',
        'address': 'http://24.83.27.62:50104'
    },
    {
        'ss58_address': '5HWQ14cXWTBa3V17x576w9rhaLH8DgWfvNQTyyhxKYzYXozs',
        'name': 'subspace.vali',
        'address': 'http://24.83.27.62:50060'
    },
    {
        'ss58_address': '5CqhADCJeHn8UV5ReoXS57bSEQftVqMd8CLcGqrhqEFKfjFm',
        'name': 'subspace.vali::tang',
        'address': 'http://24.83.27.62:50062'
    },
    {
        'ss58_address': '5CfdAcN1UmE4iNxdLa5vB8r2dDD3qQMiMMxaaVLmCc2cGWCp',
        'name': 'subspace::tang0',
        'address': 'http://24.83.27.62:50092'
    },
    {
        'ss58_address': '5D29dSg3cZPubeGPmwC9euTc5ahvRZULE42vLEyWzPQisRJE',
        'name': 'subspace::tang1',
        'address': 'http://24.83.27.62:50133'
    },
    {
        'ss58_address': '5EL8yYNyEoeVeavaW8e2KPk2ay2oUwFnFTNbqVYrMHLjMQeV',
        'name': 'subspace::tang2',
        'address': 'http://24.83.27.62:50067'
    },
    {
        'ss58_address': '5FC3SbvtQs9n6ohT897uQmRxBwiipY7VVXwcBQgz4mh2vE2X',
        'name': 'subspace::tang3',
        'address': 'http://24.83.27.62:50082'
    },
    {
        'ss58_address': '5Hj6fGoSWX4QdEcuJEH476wbnJRWq54qusiYezFqdGtKzDVu',
        'name': 'subspace::tang4',
        'address': 'http://24.83.27.62:50061'
    },
    {
        'ss58_address': '5FvHzPds25qtc5BtGcd53knX6CVTXMaeCJBhdmwrBwT9Cbm3',
        'name': 'subspace::tang5',
        'address': 'http://24.83.27.62:50087'
    },
    {
        'ss58_address': '5FS3ywQfjS9D5SD72BDLFN5GUn2zTQKSUwRi9TMYxqPdJgUr',
        'name': 'subspace::tang6',
        'address': 'http://24.83.27.62:50122'
    },
    {
        'ss58_address': '5CXzVyY66RURNh39ujV7GDM78wEN6FfUnvA23Z1mSinLtfW9',
        'name': 'subspace::tang7',
        'address': 'http://24.83.27.62:50106'
    },
    {
        'ss58_address': '5D7VUk5eWWRAk4wJByQbRffQLDPy45ErG1fa6r1yj2aFN3Dp',
        'name': 'subspace::tang8',
        'address': 'http://24.83.27.62:50120'
    },
    {
        'ss58_address': '5F7B5eWvgjF6JPPyA7hFSgcLkykTjuYRwYqqPoTH9Su7oLMf',
        'name': 'subspace::tang9',
        'address': 'http://24.83.27.62:50068'
    }
]

netuid= 0
subspace = c.m('subspace')()
launcher_keys = subspace.launcher_keys()
keys = subspace.keys(netuid=netuid, max_age=10)

fututres = []
timeout = 60

for i, batch in enumerate(batches):
    if batch['ss58_address'] in keys:
        c.print(f"Skipping {batch['name']}")
        continue
    kwargs = dict(module_key=batch['ss58_address'], 
                                        name=batch['name'], 
                                        key = launcher_keys[i % len(launcher_keys)],
                                        address=batch['address'])
    c.print(kwargs)
    f = c.submit(c.register, kwargs=kwargs, timeout=timeout)
    fututres.append(f)
for f in c.as_completed(fututres, timeout=timeout):
    print(f.result())