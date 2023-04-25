import commune
client = commune.client('162.157.13.236:9054')
print(client.forward(fn='schema'))