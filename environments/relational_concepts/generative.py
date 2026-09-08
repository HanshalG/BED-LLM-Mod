"""A declared synthetic grammar prior, not INDUCTION's private release pipeline."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import random

from concept_synth.sexpr_parser import parse_sexpr_formula
from concept_synth.fol.model import FiniteModel, evaluate

MAX_HEIGHT = 4
VARIABLES = ('x', 'y', 'z', 'u', 'v')
OPERATORS = ('atom', 'not', 'and', 'or', 'exists', 'forall')
WEIGHTS = (4, 1, 2, 2, 2, 2)


def _rng(seed, stream):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    digest = hashlib.sha256(f'relational-concepts-v1:{stream}:{seed}'.encode()).digest()
    return random.Random(int.from_bytes(digest, 'big'))


def atoms(scope):
    return (tuple(f'({p} {v})' for p in ('P', 'Q') for v in scope)
            + tuple(f'({p} {a} {b})' for p in ('R', 'S') for a in scope for b in scope)
            + tuple(f'(= {a} {b})' for i, a in enumerate(scope) for b in scope[i+1:]))


def _draw(rng, height, scope):
    op = 'atom' if height == 0 else rng.choices(OPERATORS, weights=WEIGHTS, k=1)[0]
    if op == 'atom':
        return rng.choice(atoms(scope))
    if op == 'not':
        return f'(not {_draw(rng, height-1, scope)})'
    if op in ('and', 'or'):
        return f'({op} {_draw(rng, height-1, scope)} {_draw(rng, height-1, scope)})'
    variable = VARIABLES[len(scope)]
    return f'({op} {variable} {_draw(rng, height-1, scope+(variable,))})'


@dataclass(frozen=True)
class PrivateConcept:
    formula: str
    structural_draws: int

    def label(self, world, object_index):
        if type(object_index) is not int or not 0 <= object_index < world.size:
            raise ValueError('invalid object index')
        return evaluate(parse_sexpr_formula(self.formula), world.to_model(), {'x': object_index})


def sample_concept(seed):
    rng = _rng(seed, 'concept')
    for attempt in range(1, 65):
        text = _draw(rng, MAX_HEIGHT, ('x',))
        if parse_sexpr_formula(text).free_vars() == {'x'}:
            return PrivateConcept(text, attempt)
    raise ValueError('structural sampler exhausted; do not replace the seed')


@dataclass(frozen=True)
class PublicWorld:
    size: int
    unary: tuple
    binary: tuple

    def to_model(self):
        return FiniteModel(self.size, unary={p: set(xs) for p, xs in self.unary},
                           binary={p: set(xs) for p, xs in self.binary})

    def public_payload(self):
        return dict(domain=[f'a{i}' for i in range(self.size)],
                    unary={p: [f'a{i}' for i in xs] for p, xs in self.unary},
                    binary={p: [[f'a{i}', f'a{j}'] for i, j in xs] for p, xs in self.binary})


def sample_world(seed):
    """No concept argument, label rejection, or shared concept RNG stream."""
    rng = _rng(seed, 'world')
    size = rng.randint(7, 13)
    unary, binary = [], []
    for predicate in ('P', 'Q'):
        density = rng.uniform(.2, .8)
        unary.append((predicate, tuple(i for i in range(size) if rng.random() < density)))
    for predicate in ('R', 'S'):
        density = rng.uniform(.1, .5)
        binary.append((predicate, tuple((i, j) for i in range(size) for j in range(size)
                                        if rng.random() < density)))
    return PublicWorld(size, tuple(unary), tuple(binary))


@lru_cache(maxsize=None)
def derivation_count(height, scope_size):
    """Exact ordered grammar tree count BEFORE the free-x structural condition."""
    if type(height) is not int or type(scope_size) is not int or height < 0 or scope_size < 1:
        raise ValueError('invalid grammar dimensions')
    atom_count = 2*scope_size + 2*scope_size**2 + scope_size*(scope_size-1)//2
    if height == 0:
        return atom_count
    child = derivation_count(height-1, scope_size)
    return atom_count + child + 2*child**2 + 2*derivation_count(height-1, scope_size+1)
