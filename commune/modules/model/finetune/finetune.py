import commune as c
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
from datasets import load_dataset
import torch

class ModelFinetune(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y

    def load_model(self, model_name='gpt2'):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        # Ensure the tokenizer uses the correct padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer, model

    def train_model(self, model, tokenizer, dataset, epochs=1, learning_rate=1e-5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for idx, batch in enumerate(dataset):
                inputs = tokenizer(batch["text"], return_tensors='pt').to(device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if idx % 100 == 0:
                    print(f"Epoch: {epoch}, Loss:  {loss.item()}")

    def prompt_model(self, model, tokenizer, prompt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = model.generate(inputs, max_length=100, temperature=0.7, num_return_sequences=1)
        return tokenizer.decode(outputs[0])


    def prompt(self, prompt):
        # Load the model
        tokenizer, model = self.load_model()

        # Load the dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

        # Filter out empty strings
        dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)

        # Preprocess the dataset
        dataset = dataset['train'].map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)

        # Train the model
        self.train_model(model, tokenizer, dataset)

        return self.prompt_model(model, tokenizer, prompt)

    def train(self, sentences, epochs=1, learning_rate=1e-5):
        # Load the model
        tokenizer, model = self.load_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for idx, sentence in enumerate(sentences):
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if idx % 100 == 0:
                    print(f"Epoch: {epoch}, Loss:  {loss.item()}")
                    