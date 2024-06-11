

import commune as c
class Judge(c.Module):
    def __init__(self, model='model.openai'):
        self.model = c.module(model)()

    def forward(self, text: str = "was the moon landing fake?") -> str:

        prompt = {
            "text": text,
            'question': 'Yay or nay? (Y/N)',
        }
        return self.model.forward(prompt)
    

    def test(self , text: str = "was the moon landing fake?"):
        return self.forward(text)


if __name__ == "__main__":
    
    Judge.run()





