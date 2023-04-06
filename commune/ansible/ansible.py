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
        self.load_yaml(path=self.inventory_path)
        
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
        print(self.ping())
        # cls.print(self.inventory)
        # self.mv_node('all2', 'all')
        # self.save()
        
    def ping(self):
        return self.cmd(f"ansible all -m ping -i {self.inventory_path}")
        
if __name__ == '__main__':
    Ansible.run()
    
    
