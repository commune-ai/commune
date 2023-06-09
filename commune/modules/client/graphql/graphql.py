import os
from .utils import graphql_query
from commune import Module
import ray

class GraphQLModule(Module):
    default_config_path = "client.graphql.module"

    def __init__(self,config,):
        Module.__init__(self, config)
        self.



    def url(self):
        if not hasattr(self,'_url'):
            url = self.config.get('url')
            if url == None:
                assert 'host' in self.config
                assert 'port' in self.config

            url = f"http://{config['host']}:{config['port']}"
            self._url = url
        
        return self._url

    @url.setter
    def url(self, value:str):
        self._url = value


    
    def query(self,query, url=None, return_one=False):
        output = graphql_query(url=self.url, query=query)
        if return_one:
            output = list(output.values())[0]
        
        return output


    def query_list(sef, query_list, num_actors=2, url=None):
        if url != None:
            self.url = url
        
        ray_graphql_query = ray.remote(graphql_query)
        ready_jobs = []
        for query in query_list:
            ready_jobs.append(ray_graphql_query.remote(url=self.url, query=query))
        
        finished_jobs_results  = []
        while ready_jobs:
            ready_jobs, finished_jobs = ray.wait(ready_jobs)
            finished_jobs_results.extend(ray.get(finished_jobs))

        return finished_jobs_results


