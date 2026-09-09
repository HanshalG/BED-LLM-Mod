"""Ground-fact parser and two-task source audit; does not execute Prolog."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request

from lark import Lark, Transformer

COMMIT = '33e166058404fd5eec3ec0f5080df9befbe884a8'
HASHES = {
    '1': {'bk.pl':'b3ab6398b9b054090cbfa89e3cf8874c8c01cb0aed652cdca07c8854e8408c25',
          'exs.pl':'4820860c45307f8303bb61ef0e0208b2a03dc1362a0d6e9a7c4b6c7ca6d4c31a',
          'bias.pl':'14a735756169f0a70a10a7b4cf70b07973a30f1c424651fcf86767aaa58371f2'},
    '10': {'bk.pl':'c3bb3bb0a9e713c477b7b2206578dc93618aed18461e33ffca8fc058838fe74b',
           'exs.pl':'f713fc19d682c9a9ab144c10ba5f8078747c1c275d5a378c6ef968cbaba95124',
           'bias.pl':'e6cf9f5b986f0a2dfe26e6a977669c949accee58a8c6e0c654bb35b556179afe'},
}
GRAMMAR = r"""
start: (term ".")*
?term: NAME "(" term ("," term)* ")" -> compound
     | NAME -> atom
     | QUOTED -> quoted
     | INT -> integer
NAME: /[a-z][A-Za-z0-9_]*/
QUOTED: /'(?:[^'\\]|\\.|'')*'/
INT: /-?[0-9]+/
%import common.WS
%ignore WS
"""


class Facts(Transformer):
    def start(self, xs):
        return xs

    def compound(self, xs):
        return (str(xs[0]), tuple(xs[1:]))

    def atom(self, xs):
        return str(xs[0])

    def integer(self, xs):
        return int(xs[0])

    def quoted(self, xs):
        text = str(xs[0])[1:-1]
        result, i = '', 0
        while i < len(text):
            c = text[i]
            if c == '\\':
                i += 1
                escapes = {"'":"'", '\\':'\\', 'n':'\n', 't':'\t', 'r':'\r'}
                if i >= len(text) or text[i] not in escapes:
                    raise ValueError('unsupported atom escape')
                c = escapes[text[i]]
            elif c == "'":
                if text[i:i+2] != "''":
                    raise ValueError('bad quoted atom')
                i += 1
            result += c
            i += 1
        return result


PARSER = Lark(GRAMMAR, parser='lalr', transformer=Facts())


def parse(text):
    if len(text.encode()) > 1000000:
        raise ValueError('fact byte cap')
    return PARSER.parse(text)


def inspect(bk, exs):
    inputs, outputs, widths = defaultdict(dict), defaultdict(dict), {}
    positive, negative = set(), set()
    def add(rows, args):
        e, pos, char = args
        if type(e) is not str or type(pos) is not int or pos < 1 or type(char) is not str or len(char)!=1:
            raise ValueError('invalid character coordinate')
        if pos in rows[e] and rows[e][pos] != char:
            raise ValueError('conflicting characters')
        rows[e][pos] = char
    for name,args in parse(bk):
        if name=='in':
            add(inputs,args)
        elif name=='width':
            e,w = args
            if type(w) is not int or w<0 or (e in widths and widths[e]!=w):
                raise ValueError('invalid input width')
            widths[e] = w
    for name,args in parse(exs):
        if name not in ('pos','neg') or len(args)!=1 or args[0][0]!='out':
            raise ValueError('unexpected label predicate')
        row = args[0][1]
        (positive if name=='pos' else negative).add(row)
        if name=='pos':
            add(outputs,row)
    if positive & negative:
        raise ValueError('positive/negative conflict')
    if set(inputs)!=set(widths) or set(inputs)!=set(outputs):
        raise ValueError('missing input/output/width; empty-output convention unresolved')
    def string(chars):
        if set(chars)!=set(range(1,len(chars)+1)):
            raise ValueError('noncontiguous positions')
        return ''.join(chars[i] for i in range(1,len(chars)+1))
    pairs = [(string(inputs[e]),string(outputs[e])) for e in sorted(inputs)]
    if any(len(inputs[e])!=widths[e] for e in inputs):
        raise ValueError('width does not match input')
    return dict(examples=len(inputs),distinct_inputs=len({x for x,y in pairs}),
        positive_facts=len(positive),negative_facts=len(negative),label_conflicts=0,
        contiguous_coordinates=True,complete_function_outside_examples=False,
        enough_for_one_initial_six_candidates_three_targets=len({x for x,y in pairs})>=10)


def run():
    result = dict(source_commit=COMMIT,scope='Only predesignated development tasks 1 and 10',
                  declared_card_license='mit',model_calls=0,cost_usd=0,paid_authority=False,tasks={})
    for task,files in HASHES.items():
        data = {}
        for name,sha in files.items():
            url=f'https://huggingface.co/datasets/andrewcropper/ilp-datasets/resolve/{COMMIT}/strings/{task}/train/{name}'
            with urllib.request.urlopen(url,timeout=30) as r:
                raw=r.read(1000001)
            if len(raw)>1000000 or hashlib.sha256(raw).hexdigest()!=sha:
                raise ValueError('source size/hash changed')
            data[name]=raw.decode()
        result['tasks'][task]=dict(inspect(data['bk.pl'],data['exs.pl']),sha256=files)
    return result


if __name__=='__main__':
    result=run()
    with Path('results/nonmyopic/AUTHOR_STRINGS_DEV_AUDIT_20260909.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True)
        f.write('\n')
    print(json.dumps(result,sort_keys=True))
