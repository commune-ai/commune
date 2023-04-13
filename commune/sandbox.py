from transformers import AutoTokenizer, AutoModelForCausalLM
import commune
model = 'EleutherAI/gpt-neo-125M'
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)
model = commune.module(model)
model.serve(name='gpt125m', wait_for_termination=False)
commune.new_event_loop(True)
model = commune.connect('gpt125m')

print(model.functions())