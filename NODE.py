class NODE:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def type(self):
        # Type of node: 0.operand 1.unary operator 2.binary operator
        if self.left is None:
            return 0
        elif self.right is None:
            return 1
        else:
            return 2

    def depth(self):
        if self.type() == 0:
            return 1
        else:
            if self.type() == 2 and self.right.depth() > self.left.depth():
                return self.right.depth() + 1
            else:
                return self.left.depth() + 1


