"""Native list-grid Python interface validation; container is the security boundary."""
import ast

MODULES = frozenset({'math','collections','itertools','functools','heapq'})


def validate(code):
    if not isinstance(code,str) or len(code.encode())>16384:
        raise ValueError('code size')
    tree=ast.parse(code,feature_version=8)
    if sum(1 for _ in ast.walk(tree))>4096:
        raise ValueError('code complexity')
    transforms=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transform']
    if len(transforms)!=1:
        raise ValueError('one transform function required')
    args=transforms[0].args
    if len(args.args)!=1 or args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
        raise ValueError('transform takes one grid')
    for node in ast.walk(tree):
        if isinstance(node,(ast.ClassDef,ast.AsyncFunctionDef,ast.AsyncFor,ast.AsyncWith)):
            raise ValueError('unsupported definition')
        if isinstance(node,ast.FunctionDef) and node.decorator_list:
            raise ValueError('decorators unavailable')
        if isinstance(node,ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('private attribute unavailable')
        if isinstance(node,ast.Name) and node.id.startswith('__'):
            raise ValueError('dunder name unavailable')
        if isinstance(node,ast.Import) and any(n.name not in MODULES for n in node.names):
            raise ValueError('module unavailable')
        if isinstance(node,ast.ImportFrom) and (node.level or node.module not in MODULES or any(n.name.startswith('_') or n.name=='*' for n in node.names)):
            raise ValueError('module unavailable')
    return tree
