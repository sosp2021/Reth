class Interval:
    def __init__(self, f, interval=1):
        self.cur = 0
        self.interval = interval
        self.f = f

    def call(self, step=1):
        self.cur += step
        if self.cur >= self.interval:
            self.f()
            self.cur %= self.interval

    def __call__(self, step=1):
        self.call(step)
