import commune
import openai
openai.api_key = "YOUR_API_KEY_HERE"

# Define a new class that inherits from commune.Module
class OpenAIModel(commune.Module):
    def __init__(self):
        pass
    def generate_text(self, prompt):
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text

# Launch your OpenAI model as a public server
OpenAIModel.launch(name='openai_model')

# Connect to the model and generate some text
openai_model = commune.connect('openai_model')
text = openai_model.generate_text("Hello, world!")
print(text)
