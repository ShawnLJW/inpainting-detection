class RunningAggregation:
    def __init__(self):
        self.val = 0
        self.n = 0

    def add(self, x, n=1):
        self.val += x
        self.n += n

    def __call__(self):
        if self.n > 0:
            return self.val / self.n
        return None