from abc import abstractmethod


class LearningRateScheduler:
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    def __call__(self, step: int) -> float:
        return self.learning_rate(step)

    @abstractmethod
    def learning_rate(self, step: int) -> float:
        raise NotImplementedError()
