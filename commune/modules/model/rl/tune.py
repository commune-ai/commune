import commune
try:
    from ray import tune
except ImportError:
    commune.run_command('pip install ray[tune]')
    from ray import tune

# 1. Define an objective function.
def objective(config):
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


# 2. Define a search space.
search_space = {
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3]),
}

# 3. Start a Tune run and print the best result.
tuner = tune.Tuner(objective, param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)