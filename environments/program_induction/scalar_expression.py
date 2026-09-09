"""Bounded scalar arithmetic for untrusted proposals; no eval, exec or imports."""
import ast
import math
import operator

FUNCTIONS = {name:getattr(math,name) for name in ('exp','log','sqrt','sin','cos','tan')}
BINARY = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
          ast.Div:operator.truediv, ast.Pow:math.pow}


class ScalarExpression:
    def __init__(self, text, names):
        if type(text) is not str or len(text.encode())>8192:
            raise ValueError('expression byte cap')
        self.names = frozenset(names)
        if len(self.names)>16 or any(type(n) is not str or not n.isidentifier()
                                    or n.startswith('_') or n in FUNCTIONS or n in ('np','pi')
                                    for n in self.names):
            raise ValueError('invalid variable names')
        try:
            self.tree = ast.parse(text, mode='eval').body
        except (SyntaxError,RecursionError) as error:
            raise ValueError('invalid expression') from error
        if sum(1 for _ in ast.walk(self.tree))>512:
            raise ValueError('expression node cap')
        self.validate(self.tree, 0)

    def validate(self, node, depth):
        if depth>32:
            raise ValueError('expression depth cap')
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            if not math.isfinite(float(node.value)):
                raise ValueError('nonfinite constant')
        elif isinstance(node,ast.Name) and node.id in self.names | {'pi'}:
            pass
        elif isinstance(node,ast.Attribute) and ast.unparse(node)=='np.pi':
            pass
        elif isinstance(node,ast.BinOp) and type(node.op) in BINARY:
            self.validate(node.left,depth+1)
            self.validate(node.right,depth+1)
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            self.validate(node.operand,depth+1)
        elif isinstance(node,ast.Call) and len(node.args)==1 and not node.keywords:
            name = ast.unparse(node.func)
            if name not in FUNCTIONS and name not in {'np.'+k for k in FUNCTIONS}:
                raise ValueError('function not allowed')
            self.validate(node.args[0],depth+1)
        else:
            raise ValueError('unsupported expression')

    def __call__(self, point):
        if set(point)!=self.names or any(type(v) not in (float,int) or not math.isfinite(float(v))
                                         for v in point.values()):
            raise ValueError('invalid input point')

        def evaluate(n):
            if isinstance(n,ast.Constant):
                value = float(n.value)
            elif isinstance(n,ast.Name):
                value = math.pi if n.id=='pi' else float(point[n.id])
            elif isinstance(n,ast.Attribute):
                value = math.pi
            elif isinstance(n,ast.UnaryOp):
                value = evaluate(n.operand) * (-1 if isinstance(n.op,ast.USub) else 1)
            elif isinstance(n,ast.Call):
                value = FUNCTIONS[ast.unparse(n.func).removeprefix('np.')](evaluate(n.args[0]))
            else:
                a,b = evaluate(n.left),evaluate(n.right)
                if isinstance(n.op,ast.Pow) and abs(b)>32:
                    raise ValueError('exponent cap')
                value = BINARY[type(n.op)](a,b)
            if not math.isfinite(value):
                raise ValueError('nonfinite result')
            return value
        try:
            return evaluate(self.tree)
        except (ArithmeticError,OverflowError) as error:
            raise ValueError('invalid numerical operation') from error
