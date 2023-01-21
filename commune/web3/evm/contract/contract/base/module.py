from commune import Module
import streamlit as st

class BaseContract(Module):
    def __init__(self):
        Module.__init__(self)
        self.setup_web3_env()

    def setup_web3_env(self):
        self.contract_manager = self.launch('web3.contract', kwargs=dict(network=self.config['network']))
        self.contract_manager.set_account(self.config['account'])
        st.write(self.config['contract'])
        self.contract = self.contract_manager.deploy_contract(**self.config['contract'])
        self.account = self.contract.account
        self.web3 = self.contract.web3

    def mint(self, amount=1000, account=None):

        account =self.resolve_account(account)
        self.contract.mint(account.address, amount)

        return self.balanceOf(account)

    # for the demo, the buy is the mint function
    buy = mint
    @property
    def address(self):
        return self.account.address

    def set_account(self, account):
        self.account.set_account(account)
        return self.account.address


    def add_stake(self, amount, account = None):
        account = self.resolve_account(account)
        amount = int(amount)
        self.contract.approve(self.contract.address,amount)
        self.contract.allowance(account.address, account.address)
        self.contract.add_stake(amount)
        return self.get_stake(account) 

    def remove_stake(self, amount, account = None):
        return self.contract.remove_stake(amount)

    def get_stake(self, account = None):
        account = self.resolve_account(account)
        return self.contract.get_stake(account.address)


    def set_votes(self, accounts:list, votes=100):

        for i in range(len(accounts)):
            if accounts[i] in self.peers:
                accounts[i] = self.peers[accounts[i]].address
        if isinstance(votes, int): 
            votes = [votes]*len(accounts)
        self.contract.set_votes(votes, accounts)
        return self.get_peer_score_map()


    def resolve_account(self, account=None): 
        if account == None:
            account = self.account 
        return account


    def balanceOf(self, account=None):
        account =self.resolve_account(account)
        return self.contract.balanceOf(account.address)
        

    @property
    def peers(self):
        if not hasattr(self, '_dev_accounts'):
            self._dev_accounts = {account: self.account.replicate(account) for account in ['alice', 'bob', 'chris', 'dan'] }
        return self._dev_accounts

    dev_accounts = peers

    def add_peers(self, accounts, votes=100):

        # add votes to same number of peers
        accounts = self.dev_accounts
        self.set_votes(accounts=[a.address for a in accounts], votes=votes)


        
    def get_state(self, account=None):
        account =self.resolve_account(account)
        state_tuple = self.contract.dev2state(account.address)
        return  dict(zip(['stake', 'score', 'block_number'],list(state_tuple)))



    def get_peer_state_map(self):
        peer_state_map = {}
        for name, account in self.dev_accounts.items():
            peer_state_map[name] = self.get_state(account)
        
        return peer_state_map

    def get_peer_score_map(self):
        peer_score_map = {}
        total_score = 0
        for peer, peer_state in self.get_peer_state_map().items():
            peer_score_map[peer] = peer_state['score']
            total_score += peer_state['score']
        
        for peer, peer_state in peer_score_map.items():
            peer_score_map[peer] /= (total_score + 1e-10)

        return peer_score_map
            

    @classmethod
    def streamlit_demo(cls):
        self = cls()
        amount = 10000
        self.mint(1)
        self.add_stake(1)
        # self.remove_stake(1)
        st.write(self.set_votes(['alice'], 100))
        

        st.write(self.get_peer_state_map())

    @classmethod
    def streamlit(cls):
        cls.streamlit_demo()


    @classmethod
    def gradio(cls):
        self = cls()
        import gradio 
        functions, names = [], []

        fn_map = {}


        fn_map['Stake Tokens'] = {'fn': self.add_stake, 
                        'inputs': [gradio.Slider(label='Stake Amount', minimum=1, maximum=1000 )],
                        'outputs':[gradio.Label(label='Current Stake', value=self.get_stake(), show_label=True)]}
        

        fn_map['Buy Tokens'] = {'fn': self.buy , 
                        'inputs': [gradio.Slider(label='Amount', minimum=1, maximum=1000 )],
                        'outputs':[gradio.Label(label='Current Balance', value=self.get_stake(), show_label=True)]}
        

        fn_map['Vote'] = {'fn': self.set_votes, 
                        'inputs':[gradio.CheckboxGroup(choices=[p for p in self.peers.keys()], value=[p for p in self.peers.keys()]),
                                  gradio.Slider(label='Score', minimum=0, maximum=100 )],
                        'outputs':[gradio.Label(label='Current Score Map', value=self.get_stake(), show_label=True)]}

        fn_map['Peer Score Map'] = {'fn': self.get_peer_score_map, 
                        'outputs':[gradio.Label(label='peer scores', value={}, show_label=True)]}
        

        fn_map['Set Account'] = {'fn': self.set_account, 
                        'inputs':[gradio.Dropdown(label=f'Account Name', choices=list(self.peers.keys()), value='alice')],
                        'outputs':[gradio.Textbox(label='Account Address', lines=1, placeholder=self.account.address)]}




        for fn_name, fn_obj in fn_map.items():
            inputs = fn_obj.get('inputs', [])
            outputs = fn_obj.get('outputs',[])
            fn = fn_obj['fn']
            names.append(fn_name)
            functions.append(gradio.Interface(fn=fn, inputs=inputs, outputs=outputs))
        
        return gradio.TabbedInterface(functions, names)



if __name__ == '__main__':
    BaseContract.run()
