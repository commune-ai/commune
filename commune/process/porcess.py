class Process:


    def __init__(self, name):
        self.name = name
        self.process = None
        self.processed = False
        self.result = None

    def set_process(self, process):
        self.process = process

    def run(self):
        self.result = self.process()
        self.processed = True

    def get_result(self):
        if not self.processed:
            raise Exception('Process not run')
        return self.result

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)