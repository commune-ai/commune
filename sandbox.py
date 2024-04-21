import commune as c


def register_many(key2address ,
    timeout=60,
    netuid = 0):
    futures = []
    launcher_keys = c.launcher_keys()
    future2launcher = {}
    future2module = {}
    registered_keys = c.m('subspace')().keys(netuid=netuid)
    progress = c.tqdm(total=len(key2address))
    while len(key2address) > 0:
        modules = list(key2address.keys())
        for i, module in enumerate(modules):
            module_key = key2address[module]
            if module_key in registered_keys:
                c.print(f"Skipping {module} with key {module}")
                key2address.pop(module)
                progress.update(1)
                continue
            c.print(f"Registering {module} with key {module}")
            launcher_key = launcher_keys[i % len(launcher_keys)]
            kwargs=dict(name=module, module_key=module_key, serve=True, key=launcher_key)
            future = c.submit(c.register, kwargs=kwargs, timeout=timeout)
            future2launcher[future] = launcher_key
            future2module[future] = module

        futures = list(future2launcher.keys())

        for f in c.as_completed(futures, timeout=timeout):
            module = future2module.pop(f)
            launcher_key = future2launcher.pop(f)
            module_key = key2address.pop(module)
            c.print(f"Registered {module} module_key:{module_key} launcher_key:{launcher_key}")
            r = f.result()
            if c.is_error(r):
                progress.update(1)
            
key2address = {
        'storage::fam0': '5EFkMwcoJGUVDV6zV33kak1rWSfcC6mwqxPXRmFAnE5S5EDe',
        'storage::fam1': '5G6XQ8JjvCDYt9AnR5N3aPcGVHk5VtacrZsJjszefNQ8o1Lu',
        'storage::fam10': '5Gc9SvKasskFoEzh2ugshP4P3oPuwxo8L1jhLvNz3Fwuko6z',
        'storage::fam12': '5Dtr56SvB4bCaiDC5uQxdyCpjp2UrM8V7zq9KwoUJp29zopk',
        'storage::fam16': '5CLtU7g8Fe9E64i5iYABWrTRRQ5gA3L51e1qjyWdDqgodT7A',
        'storage::fam11': '5D5WUcEazbiCH3vv4rXEKkKrbykdzzAnkUgVLu6n2EqJUmz5',
        'storage::fam2': '5Do9gQic7McHFA23E3EeUTfPVgiHrzbbhBVzMcwYYMiFkYr5',
        'storage::fam18': '5DyaQkNzsZ1jin8q7mQa2xQFZcQB2K1spe22nUoEiVtqFnpo',
        'storage::fam17': '5CtdvD8k7oEJ4j8tyFarfsRV6yNKYyybuwYwmNk56MWHRF73',
        'storage::fam14': '5Gb4itZ45siQ98nGVTPEmY2SGd2biudtPsG9NsGRnxFSgeQM',
        'storage::fam15': '5Ea2FLbV7uL2ojTYzy4hraVXq279syrJQgBxn4vYZyAAR5gP',
        'storage.vali::fam': '5DnNuTYcGAGe65ncPAQywXnzFL1X4en4hBSPZAqAXrnk2xwU',
        'storage::fam13': '5EXH1p5TW8e9zfgdouXWAaBBdwKYxRaKhoBNnEWL9V6KeHCv',
        'storage::fam19': '5EEpKVFxChWThH8vHfXYAgxnr4T4DQ8BSeZj58ptv44ypPAr',
        'storage::fam7': '5HgfutrPcLXs7c2K95TLjiQPYB9PqGLBagmDfps6CDKYu7kF',
        'storage::fam5': '5GGrnUQmoyKWwgLPFpB9XtqnGDch86KCSQbwVURWiJwnE8ux',
        'storage::fam3': '5F1wWmRRsyBamx7wzpYqWPPHGpKiyrcw9Vu4vLkbbD3Hf6Xz',
        'storage::fam6': '5DeaTJWqMUrGHUDdvYDNC51vaNh9aPis8ZSTAwviNFcwheob',
        'storage::fam4': '5H1XG6CSzcyyycxQvCDZHk1sYzxmDKv8B4KDCTGScDyjmEjw',
        'storage::fam9': '5F6qT8iD4ENMrk6Zgod7eTE5ivCCMaxs6o4ghBE47VanpM3b',
        'storage::fam8': '5CVag65XTgtQdZY3AuTfPJ94DwLUsagxQsPmLjSCLJ3fFniv'
    }

register_many(key2address=key2address)


