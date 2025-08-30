import commune as c 

class Scanner:
    def ask(self, 
            context="./",
            batch_size = 4,
            ):
        
        question = """
        is there any vulnerabilities when running this, i dont care about exploits
        but more concerned about attacks on my system that can be done by running it 
        locally
        >0.5 if its too dangerous to run locally, ONLY ANSWER IN THE JSON FORMAT
        json(score:float) [0,1] where threshold
        DO NOT INCLUDE EXTRA FIELDS, IF YOU THINK ITS TOO DANGEROUS
        SPECIFY THE LINE AND REASON CONCICELY
        """
         
        batch_context = {}
        batch_response = []
        for i, (file, text) in enumerate(c.file2text(context).items()):
            if len(batch_context) == batch_size :
                files = list(batch_context.keys())
                print(files)
                response = ''
                for ch in c.ask(question, context=batch_context): 
                    print(ch, end="")
                    response += ch
                batch_response.append(response)
                batch_context = {}
            else:
                batch_context[file] =  text
        return batch_response
        