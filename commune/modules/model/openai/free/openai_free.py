



import subprocess

import commune as c

class OpenAIFree(c.Module):
    def forward(self, text='hey', **kwargs):
        command = ['bito']
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(text.encode())
        output = stdout.decode().strip()
        return output

    talk = forward

    def ensure_env(self):
        c.cmd('sudo curl https://alpha.bito.ai/downloads/cli/install.sh -fsSL | bash', verbose=True)
    # # Example usage
    # input_prompt = input("InputText: ")
    # output = run_bito(input_prompt)
    # print(output)