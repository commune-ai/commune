class Memory:
    def __init__(self):
        self.memory = {}

    def store(self, address, value):
        self.memory[address] = value

    def load(self, address):
        return self.memory.get(address, 0)