import math


class Schedule:
    def __init__(self, func, max_steps=1):
        self.cur_step = 0
        self.max_steps = max_steps
        self.func = func

    @staticmethod
    def from_str(schedule_str):
        if isinstance(schedule_str, (int, float)):
            return Schedule(lambda _: schedule_str)

        parts = schedule_str.split(",")
        if len(parts) == 0:
            return Schedule(lambda _: float(parts[0]))
        elif len(parts) == 3:
            start, end, max_steps = parts
            method = "linear"
        elif len(parts) == 4:
            method, start, end, max_steps = parts
        else:
            raise Exception(f"Invalid schedule string {schedule_str}")

        start = float(start)
        end = float(end)
        max_steps = int(max_steps)

        if method == "linear":
            return Schedule(
                lambda step: start + (end - start) * step / max_steps, max_steps
            )
        elif method == "exp":
            return Schedule(
                lambda step: end - (end - start) * math.exp(-1 * step / max_steps),
                max_steps,
            )
        else:
            raise Exception(f"Invalid schedule method {method}")

    def step(self):
        if self.cur_step < self.max_steps:
            self.cur_step += 1
        return self.func(self.cur_step)

    def value(self, step=None):
        if step is None:
            step = self.cur_step
        else:
            step = min(step, self.max_steps)
        return self.func(step)
