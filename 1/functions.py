import random
import math

def y(x):
    random.seed(x)
    return random.random()


def option5(x):
    x1 = x ** 2 - 50
    x2 = 0.001 * x
    return x1, x2


def option6(x):
    x1 = math.tan(x);
    x2 = math.sin(x)
    return x1,x2


def option7(x):
    x1 = math.cos(x) * x**3;
    x2 = -3 * x**3 + 7
    return x1,x2