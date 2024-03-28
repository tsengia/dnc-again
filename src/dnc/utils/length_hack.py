
class LengthHackSampler:
    def __init__(self, batch_size, length):
        self.length = length
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            len = self.length() if callable(self.length) else self.length
            yield [len] * self.batch_size

    def __len__(self):
        return 0x7FFFFFFF