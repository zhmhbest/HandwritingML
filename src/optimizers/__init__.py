

class OptimizerBase:
    def __call__(self, param, param_grad, param_name, cur_loss=None):
        pass

    def step(self):
        """Increment the optimizer step counter by 1"""
        # self.cur_step += 1