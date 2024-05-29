import commune as c
access_token = "{timestamp}::{address}::{signature}"
print(c.module('ticket')().verify(ticket))

