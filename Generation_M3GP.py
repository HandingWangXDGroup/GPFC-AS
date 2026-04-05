from NODE import NODE
import numpy as np
import copy

#           Meaning
# 0-1       variable                    x[round(i*D)]
# 1-2       constent                    0-10
# 10        Negative                    -x
# 11        Reciprocal                  1/sqrt(1+x.^2)
# 12        Multiplying by 10           10*x
# 13        Square                      x^2
# 14        Square root                 sqrt(abs(x))
# 15        Absolute value              abs(x)
# 16        Rounded value               round(x)
# 17        Sine                        sin(2*pi*x)
# 18        Cosine                      cos(2*pi*x)
# 19        Logarithm                   log(1+abs(x))
# 20        Exponent                    exp(x)
# 21        Addition                    x+y
# 22        Subtraction                 x-y
# 23        Multiplication              x*y
# 24        Analytic Quotient           x/sqrt(1+y^2)

# 定义选择操作符的概率
mOperater = list(range(21, 25))
# -1、1/、10*、^2、sqrt、abs、round、sin、cos、log、exp、+、-、*、/
pOperater = np.array([1, 1, 1, 1])
# pOperater = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
pOperater = pOperater / pOperater.sum()


# 生成树
def generate_tree(minlen, maxlen, GorF=1):
    # 初始化树
    tree = NODE(1)
    # 随机树深度
    depth = np.random.randint(minlen, maxlen + 1, 1)[0]

    # 随机生成树,通过Ramped half-and-half方法
    # grow
    if GorF == 0:
        Grow_tree(tree, depth)
    else:
        Full_tree(tree, depth)
    return tree


def Grow_tree(tree, depth):
    while tree.depth() < depth:
        p = tree
        # 随机选择一个叶子节点
        while p.type() > 0:
            if p.type() == 1 or np.random.random() < 0.5:
                p = p.left
            else:
                p = p.right

        # 随机选择一个操作符
        operater = np.random.choice(mOperater, 1, False, pOperater)[0]
        if operater >20:  # 二元运算符
            if np.random.randint(0, 61, 1)[0] != 1:
                operand = np.random.random()  # 决策变量
            else:
                operand = 1 + np.random.random()  # 常数
            if np.random.random() < 0.5:
                p.left = NODE(p.value)
                p.right = NODE(operand)
            else:
                p.left = NODE(operand)
                p.right = NODE(p.value)
        else:
            p.left = NODE(p.value)
        p.value = operater


def Full_tree(tree, depth):
    if depth <= 1:
        if np.random.randint(0, 61, 1)[0] != 1:
            operand = np.random.random()  # 决策变量
        else:
            operand = 1 + np.random.random()  # 常数
        tree.value = operand
    else:
        operater = np.random.choice(mOperater, 1, False, pOperater)[0]
        tree.value = operater
        if operater <= 20:  # 一元运算符
            tree.left = NODE(1)
            Full_tree(tree.left, depth - 1)
        else: # 二元运算符
            tree.left = NODE(1)
            tree.right = NODE(1)
            Full_tree(tree.left, depth - 1)
            Full_tree(tree.right, depth - 1)



# 树转化为逆波兰式
def tree2expr(tree):
    if tree.type() == 0:
        return [tree.value]
    elif tree.type() == 1:
        return tree2expr(tree.left) + [tree.value]
    else:
        return tree2expr(tree.left) + tree2expr(tree.right) + [tree.value]


def expr2func(expr, D=10, Popcal=False):
    # Popcal表示是否用于numpy进行整个种群的运算，D表示维度

    func = []
    for i in expr:
        if i < 1:  # Real number in 1-10
            if Popcal:
                func += [f"x[:,{int(i * D)}]"]
            else:
                func += [f"x[{int(i * D)}]"]
        elif i < 2:  # const
            func += [f"{(i - 1)}"]
        elif i == 21:  # Addition
            func = func[:-2] + ['({}+{})'.format(func[-2], func[-1])]
        elif i == 22:  # Subtraction
            func = func[:-2] + ['({}-{})'.format(func[-2], func[-1])]
        elif i == 23:  # Multiplication
            func = func[:-2] + ['({}*{})'.format(func[-2], func[-1])]
        elif i == 24:  # AQ
            func = func[:-2] + ['({}/np.sqrt(1+{}**2))'.format(func[-2], func[-1])]
        elif i == 10:  # Negative
            func = func[:-1] + ["(-1*{})".format(func[-1])]
        elif i == 11:  # Reciprocal
            func = func[:-1] + ["(1/np.sqrt(1+{}**2))".format(func[-1])]
        elif i == 12:  # Multiplying by 10
            func = func[:-1] + ["(10*{})".format(func[-1])]
        elif i == 13:  # Square
            func = func[:-1] + ["({}**2)".format(func[-1])]
        elif i == 14:  # Square root
            func = func[:-1] + ["(abs({})**0.5)".format(func[-1])]
        elif i == 15:  # Absolute value
            func = func[:-1] + ["(abs({}))".format(func[-1])]
        elif i == 16:  # Rounded value
            func = func[:-1] + ["(np.round({}))".format(func[-1])]
        elif i == 17:  # Sine
            func = func[:-1] + ["(np.sin(2*np.pi*{}))".format(func[-1])]
        elif i == 18:  # Cosine
            func = func[:-1] + ["(np.cos(2*np.pi*{}))".format(func[-1])]
        elif i == 19:  # Logarithm
            func = func[:-1] + ["(np.log(1+abs({})))".format(func[-1])]
        elif i == 20:  # Exponent
            func = func[:-1] + ["(np.exp({}))".format(func[-1])]
    return func[0]


def crossover(tree1, tree2):
    # 普通的交叉
    subtree1 = random_subtree(tree1)
    subtree2 = random_subtree(tree2)
    temp = copy.deepcopy(subtree1)
    subtree1.value = subtree2.value
    subtree1.left = subtree2.left
    subtree1.right = subtree2.right
    subtree2.value = temp.value
    subtree2.left = temp.left
    subtree2.right = temp.right


def mutation(tree):
    # 变异 随机选择一个
    subtree = random_subtree(tree)
    newtree=generate_tree(subtree.depth(),subtree.depth())
    subtree.value=newtree.value
    subtree.left = newtree.left
    subtree.right = newtree.right

# 随机选择一个子树
def random_subtree(tree):
    while True:
        if np.random.random() < 1 / len(tree2expr(tree)):
            return tree
        else:
            # 这里是如果没有选根节点，那么该从哪个子节点进行下一步判断
            if tree.type() == 2 and (np.random.random() < len(tree2expr(tree.right))/(len(tree2expr(tree))-1)):
                tree = tree.right
            else:
                tree = tree.left


# 随机选择一个叶子节点
def random_leaf(tree):
    p = tree
    while p.type() > 0:
        if p.type() == 1 or np.random.random() < (np.sum(np.array(tree2expr(p.left))<=10)/np.sum(np.array(tree2expr(p))<=10)):
            p = p.left
        else:
            p = p.right
    return p

def tree_edit_distance(tree1, tree2):
    if tree1 is None and tree2 is None:
        return 0

    if tree1 is None:
        return 1+tree_edit_distance(tree1, tree2.left) + tree_edit_distance(tree1, tree2.right)

    if tree2 is None:
        return 1+tree_edit_distance(tree1.left, tree2) + tree_edit_distance(tree1.right, tree2)

    if tree1.value == tree2.value:
        return tree_edit_distance(tree1.left, tree2.left) + tree_edit_distance(tree1.right, tree2.right)
    else:
        return 1+tree_edit_distance(tree1.left, tree2.left) + tree_edit_distance(tree1.right, tree2.right)


if __name__ == '__main__':
    # debug tree_edit_distance
    tree = generate_tree(5, 8)
    expr=tree2expr(tree)
    print(expr2func(expr))

    # tree=generate_tree(2,3)
    # print('mutation')
    # print(tree2expr(tree))
    # mutation(tree)
    # print(tree2expr(tree))
    #
    # print('crossover')
    # tree1=generate_tree(2,3)
    # tree2=generate_tree(2,3)
    # print(tree2expr(tree1))
    # print(tree2expr(tree2))
    # crossover(tree1,tree2)
    # print(tree2expr(tree1))
    # print(tree2expr(tree2))