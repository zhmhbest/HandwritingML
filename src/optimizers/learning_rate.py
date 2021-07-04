import numpy as np
from frame.LearningRateScheduler import LearningRateScheduler


class ConstantLearningRateScheduler(LearningRateScheduler):
    def __init__(self, lr=0.01):
        super(ConstantLearningRateScheduler, self).__init__(
            lr=lr,
            name="ConstantLearningRate"
        )

    def __str__(self):
        return f"{self['name']}(lr={self['lr']})"

    def learning_rate(self, step):
        return self['lr']


class ExponentialLearningRateScheduler(LearningRateScheduler):
    def __init__(self, initial_lr: float = 0.01, stage_length: int = 500, staircase: bool = False, decay: float = 0.1):
        """
        指数衰减法

        :param initial_lr: 初始学习率
        :param stage_length: 每个阶段的长度（单位为步）
        :param staircase: 是否仅在阶段转换时调整学习速率
        :param decay: 每一阶段学习率衰减量
        """
        super(ExponentialLearningRateScheduler, self).__init__(
            initial_lr=initial_lr,
            stage_length=stage_length,
            staircase=staircase,
            decay=decay,
            name="ExponentialLearningRate"
        )

    def __str__(self):
        return f"{self['name']}(" \
               f"initial_lr={self['initial_lr']}, " \
               f"stage_length={self['stage_length']}, " \
               f"staircase={self['staircase']}, " \
               f"decay={self['decay']})"

    def learning_rate(self, step):
        current_stage = step / self['stage_length']
        if self['staircase']:
            current_stage = np.floor(current_stage)
        return self['initial_lr'] * self['decay'] ** current_stage


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    START_LR = 0.1
    TEST_LENGTH = 500
    STAGE_LENGTH = 20

    c_lr = ConstantLearningRateScheduler(lr=START_LR)
    plt.plot([c_lr(i) for i in range(TEST_LENGTH)], label=str(c_lr))

    for j in range(8, 0, -3):
        current_decay = j / 10

        e_lr1 = ExponentialLearningRateScheduler(
            initial_lr=START_LR, stage_length=STAGE_LENGTH, decay=current_decay, staircase=False)
        e_lr2 = ExponentialLearningRateScheduler(
            initial_lr=START_LR, stage_length=STAGE_LENGTH, decay=current_decay, staircase=True)

        plt.plot([e_lr1(i) for i in range(TEST_LENGTH)], label=str(e_lr1))
        plt.plot([e_lr2(i) for i in range(TEST_LENGTH)], label=str(e_lr2))

    plt.grid()
    plt.legend()
    plt.show()
