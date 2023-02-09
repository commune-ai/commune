import commune
model_name = 'TransformerModel::EleutherAI_gpt-neo-125M'
model = commune.connect(model_name)
print(model.tokenize(['yo whadup']))
print(model.forward(['yo whadup']))