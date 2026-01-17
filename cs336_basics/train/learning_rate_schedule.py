import math


class LearningRateSchedule:
    def __init__(self,
            max_learning_rate: float,
            min_learning_rate: float,
            warmup_iters: int,
            cosine_cycle_iters: int):

        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def get_lr(self, it: int) -> float:
        if it < self.warmup_iters:
            return self.max_learning_rate * (it / self.warmup_iters)
        elif it < self.cosine_cycle_iters:

            a = it - self.warmup_iters
            b = self.cosine_cycle_iters - self.warmup_iters
            cosine_decay = 0.5 * (1 + math.cos(math.pi * a / b) )
            decayed = (self.max_learning_rate - self.min_learning_rate) * cosine_decay + self.min_learning_rate
            return decayed
        else:
            return self.min_learning_rate