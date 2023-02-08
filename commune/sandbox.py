import commune

Account = commune.get_module('commune.web3.evm.account.account.AccountModule')
Account().serve()
