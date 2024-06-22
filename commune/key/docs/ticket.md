
In the ticket the timestamp is taken, and the seperator is "::ticket::".

such that the format is 
timestamp::ticket::signature

by calling 

c.ticket(key="alice", data="alice")

the alice key signs the current timestamp and returns the ticket.


1713500654.659339::signature=e0559b535129037a62947c65af35f17c50d29b4a5c31df86b069d8ada5bcbb230f4c1e996393e6721f78d88f9b512b
6493b5ca743d027091585366875c6bea8e

now to verify the ticket you can do so like this.

c.verify_ticket("1713500654.659339::ticket::e0559b535129037a62947c65af35f17c50d29b4a5c31df86b069d8ada5bcbb230f4c1e996393e6721f78d88f9b512b6493b5ca743d027091585366875c6bea8e")

to get the signer

c.ticket2signer("1713500654.659339::ticket::e0559b535129037a62947c65af35f17c50d29b4a5c31df86b069d8ada5bcbb230f4c1e996393e6721f78d88f9b512b6493b5ca743d027091585366875c6bea8e")


To create a temperary token you can do so like this.



Temporary Tokens using Time Stampted Signaturs: Verification Without Your Keys

This allows for anyone to sign a timestamp, and vendors can verify the signature. This does not require the seed to be exposed, and can be used to identify key likley to be the same person. The only issue is if the staleness of the timestamp is too old. This can be adjusted by the vendor.


