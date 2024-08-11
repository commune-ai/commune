import commune as c


def flocktalk(self, *text, 
                search=None,
                history=None, 
                max_tokens=4000, 
                timeout=10,
                **kwargs):
    models = self.models(search=search)
    future2model = {}
    for model in models:
        kwargs = dict(text=' '.join(text), model=model, history=history, max_tokens=max_tokens)
        f = c.submit(self.forward,  kwargs=kwargs, timeout=timeout)
        future2model[f] = model
    model2response = {}
    futures = list(future2model.keys())
    try:
        for f in c.as_completed(futures, timeout=timeout):
            model = future2model[f]
            model2response[model] = f.result()
            c.print(f"{model}", color='yellow')
            c.print(f"{model2response[model]}", color='green')
    except Exception as e:
        c.print(f"Error: {e}", color='red')
    return model2response

