import commune
from ray import tune
from typing import *
import torch




class Trainer(commune.Module):
    def __init__(self, 
                 model: str = 'model.adapter', 
                 tag : str ='base',
                 metrics_server: str = 'metrics_server' ,
                 tuner: Dict = dict(
                     metric = 'loss',
                     mode = 'min',
                     max_concurrent_trials = 1,
                     resources_per_trial = {"cpu": 1, "gpu": 0},
                     num_samples= 100
                 ),
                 **kwargs):
        self.set_tag(tag)
        self.set_model(model, **kwargs)
        
        # setup the metrics serfver
        self.set_metrics_server(metrics_server)
        
        # setup the tune
        self.set_tuner(**tuner)
    
        
    
    def set_model(self, model:str, **kwargs):
        model_module = self.get_module(model)
        self.model = f'trainer::{model}'
        
        if not self.server_exists(self.model):
            model_module.launch(name=self.model, **kwargs)
        
        while not self.server_exists(self.model):
            self.sleep(1)
            self.log(f'waiting for {self.model} to register')
        
        # self.model = self.connect(self.model_name)
        
    def set_metrics_server(self, 
                          metrics_server:str,
                          refresh:bool = True,
                          timeout:int = 10,
                          check_step:int= 2):
        # if not self.server_exists(metrics_server):
        commune.launch(module='commune.metric.MetricServer', name=metrics_server, refresh=refresh)
        
        wait_time = 0

        while not self.server_exists(metrics_server) and wait_time <= timeout:
            self.sleep(check_step)
            wait_time += check_step
            
        if wait_time >= timeout:
            raise Exception('Timeout')
        self.metrics_server = metrics_server
        # self.metrics_server = self.connect(metrics_server)
        metrics_module = commune.connect(metrics_server)
        metrics_module.refresh()
        metrics_module.set_metric('baseline', 3)
        
    def hyper2params(self, params: Dict) -> Dict:
        return self.flat2deep(params)
   
   
   

    def get_hyperopt_tag(self, config:dict):
        
        tag = f'{self.model}::{self.tag}__'
        for k, v in config.items():
            tag += f'{k}_{v}__'
                
        return tag
 
 
                
    def objective(self, 
                  params:dict = {'optimizer.lr': 1e-4, 'adapter.num_layers': 2, 'adapter.hidden_dim': 3}, 
                  train_kwargs = {'num_batches': 100, 'refresh': True, 'save': True},
                  timeout:int=100) -> Dict:

        train_kwargs['tag'] = self.get_hyperopt_tag(params)
        train_kwargs['params'] = self.hyper2params(params)
        train_kwargs['metric_server'] = self.metrics_server
        train_kwargs['params']['metrics'] = {}
        train_kwargs['best_tag'] = train_kwargs['tag'].split('__')[0] + '__' + 'best'
        metrics_server = self.connect(self.metrics_server)
        model = self.connect(self.model)
        train_stats = model.train_model(**train_kwargs, timeout=timeout)
        print(train_stats['metrics'])
        return train_stats.get('metrics', 1000)
    @classmethod
    def default_search_space(cls):
        search_space = {
            'optimizer.lr': tune.loguniform(1e-4, 1e-2),
            "adapter.hidden_dim": tune.choice([32, 64, 128, 256, 512]),
            'adapter.num_layers': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        }
        
        return search_space
    
    def set_tuner(self, 
                 resources_per_trial:Dict,
                 max_concurrent_trials:int = 4,
                 num_samples: int = 10,
                 search_space:Dict = None,
                 metric: str = 'loss',
                 mode : str = 'min'):
        
        
        # 2. Define a search space.
        
        self.metric = metric
        self.mode = mode

        self.search_space = search_space if search_space else self.default_search_space()
        
        
            
        self.resources_per_trial = resources_per_trial
        self.objective_with_resources = tune.with_resources(self.objective, self.resources_per_trial)

        self.tune_config = tune.TuneConfig(num_samples=num_samples,
                                           max_concurrent_trials=max_concurrent_trials)
        
        # 3. Start a Tune run and print the best result.
        self.tuner = tune.Tuner(self.objective_with_resources, 
                                param_space=self.search_space, 
                                tune_config=self.tune_config)
    def fit(self, **kwargs):

        results = self.tuner.fit()
        print(results.get_best_result(metric=self.metric, mode=self.mode).config)


if __name__ == "__main__":
    
    # dataset = commune.connect('dataset::bittensor')
    print(Trainer().fit())
    # print(dataset.module_id)
    # for i in range(10):
    #     print('Alo')
    #     AdapterModel.train(num_batches=1, dataset=dataset)
    #     adapter = dict(
    #                 module='commune.model.adapter.block.AdapterBlock', 
    #                 params = {'in_dim': 10, 'hidden_dim': 64,  'num_layers': 8},
    #                 key2attr = {'in_dim': 'hidden_dim', 'out_dim': 'vocab_size'},
    #                 device = None
    #                 )
    # AdapterModel.run()
    # EnsembleModel.run_neuron()
    # AdapterModel.serve_module(wait_for_termination=False)
    # AdapterModel.run()
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()


