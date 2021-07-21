from abc import abstractmethod
from math import floor


class LearningRateScheduler:
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __call__(self, step: int) -> float:
        return self.learning_rate(step)

    @abstractmethod
    def learning_rate(self, step: int) -> float:
        raise NotImplementedError


class ConstantLRS(LearningRateScheduler):
    """
        固定学习率
            :param lr: 固定的学习率
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def __str__(self) -> str:
        return f"{ConstantLRS.__name__}(lr={self.lr})"

    def learning_rate(self, step: int) -> float:
        return self.lr


class ExponentialLRS(LearningRateScheduler):
    """
        指数衰减法
            :param lr: 初始学习率
            :param stage: 每个阶段的长度（单位为步）
            :param decay: 每一阶段学习率衰减量
            :param staircase: 是否仅在阶段转换时调整学习速率
    """
    def __init__(self, lr: float = 0.01, stage: int = 500, decay: float = 0.1, staircase: bool = False):
        self.lr = lr
        self.stage = stage
        self.decay = decay
        self.staircase = staircase

    def __str__(self) -> str:
        return f"{ExponentialLRS.__name__}" \
               f"(lr={self.lr}, stage={self.stage}, decay={self.decay}, staircase={self.staircase})"

    def learning_rate(self, step: int) -> float:
        stage_index = step / self.stage
        if self.staircase:
            stage_index = floor(stage_index)
        return self.lr * self.decay ** stage_index


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    START_LR = 0.1
    TEST_LENGTH = 500
    STAGE_LENGTH = 20

    c_lr = ConstantLRS(lr=START_LR)
    plt.plot([c_lr(i) for i in range(TEST_LENGTH)], label=str(c_lr))

    for j in range(8, 0, -3):
        current_decay = j / 10

        e_lr1 = ExponentialLRS(
            lr=START_LR, stage=STAGE_LENGTH, decay=current_decay, staircase=False)
        e_lr2 = ExponentialLRS(
            lr=START_LR, stage=STAGE_LENGTH, decay=current_decay, staircase=True)

        plt.plot([e_lr1(i) for i in range(TEST_LENGTH)], label=str(e_lr1))
        plt.plot([e_lr2(i) for i in range(TEST_LENGTH)], label=str(e_lr2))

    plt.grid()
    plt.legend()
    plt.show()
