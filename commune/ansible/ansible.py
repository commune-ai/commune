import commune
import os

class Ansible(commune.Module): 
    def __init__(self,  
                 inventory_path: str=None,
                 playbook_path: str=None, 
                 ):
        self.set_config()
        self.set_inventory(inventory_path)
        self.set_playbook(playbook_path)
        

        
        
    def flatten_inventory(self, inventory: dict, prefix: str = '', inventory_list: list = None):
        inventory_list = inventory_list if inventory_list != None else []
        for k,v in inventory.items():
            if isinstance(v, dict):
                if 'ansible_host' in v:
                    inventory_list.append(prefix+k)
                else:
                    self.flatten_inventory(v, prefix=prefix+k+'.', inventory_list=inventory_list)
            else:
                inventory_list.append(prefix+k)
                
        return inventory_list
    @classmethod
    def inventory_list(self):
        return self.print(self.flatten_inventory(self.inventory))
        
    def cp_node(self, from_node: str, to_node: str):
        '''
        Copy Node
        
        '''
        assert self.dict_has(self.inventory, from_node), f"from_node: {from_node} not in inventory"
        assert not self.dict_has(self.inventory, to_node), f"to_node: {to_node} already in inventory"
        self.print(f"mv_node: from_node: {from_node} to_node: {to_node}")
        from_node_data = self.copy(self.dict_get(self.inventory, from_node))
        to_node_data = self.copy(self.dict_get(self.inventory, to_node))
        self.dict_put(self.inventory, to_node, from_node_data)
        self.dict_delete(self.inventory, from_node)
        
    def mv_node(self, from_node: str, to_node: str):
        '''
        
        Move Node
        '''
        self.cp_node(from_node, to_node)
        self.dict_delete(self.inventory, from_node)

    def save(self):
        self.save_yaml(path=self.inventory_path, data=self.inventory)
        self.save_yaml(path=self.playbook_path, data=self.plays)
    def set_inventory(self, inventory_path: str=None):
        self.inventory_path = inventory_path if inventory_path!= None else self.dirpath()+'/inventory.yaml'
        self.inventory = self.load_yaml(path=self.inventory_path)
        
    def set_playbook(self, playbook_path: str=None):
        self.playbook_path = playbook_path if playbook_path else self.dirpath()+'/playbook'
        
        self.play_paths = self.glob(self.playbook_path+'/**')
        
        self.play_paths = self.glob(self.playbook_path+'/**')
 
        self.playbook = {}
        for play_path in self.play_paths:
            play_name = os.path.basename(play_path).split('.')[0]
            try:
                self.playbook[play_name] = self.load_yaml(play_path)
            except Exception as e:
                self.print(play_name)
                continue
                
        
        self.plays = list(self.playbook.keys())

    @classmethod
    def sandbox(cls):
        self = cls()
        self.print(self.shell('cd commune && make pull'), color='green')
        # cls.print(self.inventory)
        # self.mv_node('all2', 'all')
        # self.save()
        
        
        
    def play(self, play_name: str):
        return self.cmd(f"ansible-playbook {self.playbook_path}/{play_name}.yaml -i {self.inventory_path}")
    def ping(self):
        return self.cmd(f"ansible all -m ping -i {self.inventory_path}")
    
    @property
    def inventory_groups(self) -> list:
        return list(self.inventory.keys())
    def resolve_inventory(self, inventory_group: str) -> str:
        if inventory_group == None:
            inventory_group = self.inventory_groups[0]
        return inventory_group
    
    def shell(self, args:str ,
              inventory_group  : str = None, 
              chdir:str="commune", 
              verbose:bool = True,
              return_output:bool = False) -> dict:
        inventory_group = self.resolve_inventory(inventory_group)
        command = f'ansible {inventory_group} -i {self.inventory_path} -m shell -a "cd {chdir}; {args}"'
        output_text = self.cmd(command, output_text=True)
        node_chunks = output_text.split('>>')
        self.print(node_chunks)
        node2stdout = {}
        for i, node_chunk in enumerate(node_chunks):
            if i == 0:
                node_name = node_chunk.split('|')[0].strip()
            else:
                node_name = node_chunks[i-1].split('\n')[-1].split('|')[0].strip()
            if i < len(node_chunks)-1:
                node_chunk = '\n'.join(node_chunk.split('\n')[:-1])
            node2stdout[node_name] = node_chunk
        if verbose:
            for node_name, stdout in node2stdout.items():
                self.print(f"\n\n[purple bold]NODE[/purple bold]: [cyan bold]{node_name} [/cyan bold]\n")
                self.print(stdout, color='green')
        if return_output: 
    
            return node2stdout
        
    @classmethod
    def sandbox(cls, ):
        self = cls()
        cls.print(self.shell('ls'))
if __name__ == '__main__':
    Ansible.run()
    
    
