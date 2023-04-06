import commune
import os

class Ansible(commune.Module): 
    def __init__(self,  
                 inventory_path: str=None,
                 playbook_path: str=None, 
                 ):
        self.set_config()
        self.print(self.config)
        self.inventory_path = inventory_path if inventory_path!= None else self.dirpath()+'/inventory.yaml'
        self.playbook_path = self.dirpath()+'/playbook'
        self.play_paths = self.glob(self.playbook_path+'/**')
        self.play2path = {os.path.basename(play_path).split('.')[0]: play_path for play_path in self.play_paths}
        self.plays = list(self.play2path.keys())
        self.inventory = self.load_yaml(self.inventory_path)
        self.print(self.inventory)
        self.mv_node('all', 'all2')
        self.print(self.inventory)

        
        
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
        assert self.dict_has(self.inventory, from_node), f"from_node: {from_node} not in inventory"
        assert not self.dict_has(self.inventory, to_node), f"to_node: {to_node} already in inventory"
        self.print(f"mv_node: from_node: {from_node} to_node: {to_node}")
        from_node_data = self.copy(self.dict_get(self.inventory, from_node))
        to_node_data = self.copy(self.dict_get(self.inventory, to_node))
        self.dict_put(self.inventory, to_node, from_node_data)
        self.dict_delete(self.inventory, from_node)
    def mv_node(self, from_node: str, to_node: str):
        self.cp_node(from_node, to_node)
        self.dict_delete(self.inventory, from_node)



if __name__ == '__main__':
    commune.print(commune.parse_args())
    # Ansible.run()
