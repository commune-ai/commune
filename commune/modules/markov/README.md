Module Basics

a module is a set of functions with a state
let fns be the functions and the state be its state

what is a state?
a state is a set of variables that are stored locally bay the module

what is a function?
a function is an operation on params (ie. inputs to the function) to provide you with outputs
Each module has its own identity key

What is a key?
a key in commune can sign, verify, encrypt and decrypt (via AES256)

params: dict/kwargs —> module/fn
c {mod} {fn} **params (*args works too btw)

if you want to call the root module (commune/module.py)
then you can just do
c {fn} **params
for instance lets go through some examples
to generate a key
c get_key fam
<Key(address=5Gs51yMcz3orjBEVK3D8K8tLNNHw23KHZHNGHkzD9KawdcQc crypto_type=sr25519>


Client

A client calls a module by specifying the following

url  -> location of the module
fn -> function of the module
params -> params of the module 
timestamp ->

These three pieces of information are signed by the client's key and are stored in the headers while the params are stored in the json rpc call.


URL -> {ip}:{port}/{fn}

DATA  ->
{
    **params (dict)
}


The auth blob is used to verify the identity of the client and server respectively. 

HEADER (auth blob) ->

{
    data : sha256({url:url, fn:fn, params:params, time:time}) (str) 
    key: key address of the signer (str)
    signature: signature of the data with key (str)
    time: utc timestamp (str)
    max_usage: float of maximum usage the client is willing to spend
}

The request is sent to the server, calls the function is called with the params (key word values) and gets the result and the server produces a transaction reciept as follows 

transaction

the transaction is a reciept that involves the client claling

{
    fn:fn (str)
    params: params (dict)
    cost: float of cost of the transaction
    result: json serializable result (dict)
    client: headers used from the client request 
    server: a signature of the fn, params, result and client dictionary as displayed in the headers above
}


These transactions are saved locally and are posted once every epoch where the transactions are counted on chain and the cost from the transactions debits the client key. and credits the server This means the client needs to reserve tokens to as credit to be transfered to the server. 


the server defines the cost per call while the user sets a maximum usage/cost in their request. This prevents the server from maliciously setting an arbitrary cost that can exceed the agreed upon amount or advertised amount. 

This bundles trandactions on offchain and verifies onchain once every epoch to allow for scalability of verifying offchain transactions




Tree

A module can have its own network that can exist off chain or onchain. The onchain component is a permissioned list of links between 
the paranet module children modules. The parent module can link children modules as an onchain tree or network of modules that are helping the parent module. 


Link 

A link is a link between two modules that involves a profit share of the revenue between the parent and child as well as a link of the tree. A link can be made only when both modules initiate or accept the link with the proper conditions. Each link has a profit share percentage that represents the profit share between the child and the parent link. These links are unidirectional and form the structure of the tree. Each module has maxmimum set of links and can have aliases for these links as well. Links can also store metadata that contain offchain networks for more scalability. 


A Parent Child Relationship involves a parent or child initalizing a link with a child. If both parties accept, then the link is formed, and is dissolved when either party discontinues the link. This simple dual mechanism forms the backbone for forming any network/swarm/subnet. 

example of module trees

replica set (homogenous network)
a replica set is a set of modules where the children serve the same function as the parent but are offloaded from the parent to the children. 
this means the parent acts as a router and the children act as a a replica. The parent 

subnet (hetrogenous network)

a subnet is a system where the children are competing for the parent's requests. this is similar to a replica set but does not requere all of the children to be the same. This means that the subnet produces responses that are more unique and are non-deterministic as opposed to the replica set . 



recursive trees (degrees of more than 1 )

Trees can be the simple architecture of allowing for each module to have its own tree or permissioned network  


profit sharing links

When yo

consensus modules
a consensus module is a mechanism that converts

stake time tokens
when you stake to



consensus modules 

each module haas a consensu module that is is bound to. the default consensu mechanism is consensus 0 which is responsible for handling the lock and escrow for debiting and crediting offchain transacitons. we plan to allow for modules to addopt custom consensus types to fit their sepcific needs, like if they are another token on another chain or want something really specific that hasnt been done before. ideally we will allow for an offchain consensus mechanism that prooves logic consensys via a zero knowledge proof. 