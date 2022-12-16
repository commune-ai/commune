

import os
import sys
sys.path[0] = os.environ['PWD']

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient
import datetime
from streamlit_agraph import agraph, Node, Edge, Config
# **kwargs e.g. node_size=1000 or node_color="blue"
                    

class DagModule:
    nodes = []
    edges = []
    graph = []

    Node = Node 
    Edge = Edge

    config = dict(
            width=800, 
            height=500, 
            directed=True,
            nodeHighlightBehavior=True, 
            highlightColor="#F7A7A6", # or "blue"
            collapsible=True,
            node={'labelProperty':'label'},
            link={'labelProperty': 'label', 'renderLabel': True},
            update_threshold = 60
        )

    last_update_timestamp:int = datetime.datetime.utcnow().timestamp()
    def __init__(self,  config=None):
        if isinstance(config, dict):
            self.config.update(config)

        self.update_threshold = self.config['update_threshold']

    @property
    def time_since_last_update(self):
        return datetime.datetime.utcnow().timestamp()- self.last_update_timestamp

    def add_node(self, node=None, id:str=None, size:int=None, update=False ,**kwargs):

        
        if isinstance(node, Node):
            node = node
        elif isinstance(node, dict):
            node = Node(**node)
        elif node is None:
            node = Node(id=id, size=size, **kwargs)

        self.nodes.append( node)

        if update:
            self.run()
    def add_edge(self, edge=None, source:str=None, target:str=None, label:str=None, type:str='CURVE_SMOOTH', update=False,  **kwargs):
        

        if isinstance(edge, Edge):
            edge = edge
        elif isinstance(edge, dict):
            edge = Edge(**edge)

        elif edge is None:
            edge = Edge(source=source, 
                                    label=label, 
                                    target=target, 
                                    type=type)

        self.edges.append(edge)

        if update:
            self.run()

    def add_nodes(self,nodes:list, update=True):
        for node in nodes:
            node['update'] = False
            self.add_node(node)
    
        if update:
            self.run()
        
    def add_edges(self,edges:list, update=True):
        for edge in edges:
            self.add_edge(edge)
        
        if update:
            self.run()
    
    def run(self,nodes:list=None, edges:list=[]):

        
        if isinstance(nodes, list):
            self.add_nodes(nodes=nodes, update=False)
        if isinstance(edges,list):
            self.add_edges(edges=edges, update=False)


        self.graph = agraph(nodes=self.nodes, 
                edges=self.edges, 
                config=Config(**self.config))




        self.last_update_timesamp = datetime.datetime.utcnow().timestamp()

        return self.graph

    def build(self,*args, **kwargs):
        return  self.run(*args, **kwargs)
    def render(self,*args, **kwargs):
        return  self.run(*args, **kwargs)


    def add_graph(self,*args, **kwargs):
        return  self.run(*args, **kwargs)

    # @staticmethod
    # def st_example():
    #     import streamlit as st

    #     nodes = [
    #         dict(id="Spiderman", 
    #                     label="Peter Parker", 
    #                     color='blue',
    #                     size=600),

    #         dict(id="Superman", 
    #                     label="Peter Parker", 
    #                     color='green',
    #                     size=600),
    #         dict(id="Captain_Marvel", 
    #                     size=400, 
    #                     svg="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png") 
                        
    #     ]


    #     edges = [dict(source="Captain_Marvel", target="Spiderman"), dict(source="Superman", target="Spiderman")]



    #     dag = DagModule()

    #     dag.add_graph(nodes=nodes, edges=edges)

    @staticmethod
    def st_example():
        import torch
        n= 100
        edge_density = 0.05
        nodes=[dict(id=f"uid-{i}", 
                                label=f"uid-{i}", 
                                color='red',
                                size=100) for i in range(n)]

        st.write('bro')
        edges = torch.nonzero(torch.rand(n,n)>edge_density).tolist()


        edges = [(f'uid{i}',f'uid{j}') for i,j in edges]

        dag = DagModule()

