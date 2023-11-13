class Caption:
    def __init__(self, data: str):
        self.data = data

    def __str__(self):
        return self.data

    def tokenize(self):
        return self.data.split(" ")