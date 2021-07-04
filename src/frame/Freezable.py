

class Freezable:
    def __init__(self):
        self.is_freeze = False

    def freeze(self):
        """冻结"""
        self.is_freeze = True

    def unfreeze(self):
        """解冻"""
        self.is_freeze = False
