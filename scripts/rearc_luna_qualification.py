"""One-shot frozen Luna-medium qualification, with offline prefix replay."""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path

from scripts.paid_program_probe import PaidProbe, save, money, execute, route
from scripts.rearc_examples import Examples
from scripts.rearc_graph_runtime import execute as execute_graph
from scripts.rearc_symbolic_runtime import execute_symbolic
from scripts.rearc_qualification_panel import collect

ROOT = Path('results/nonmyopic/rearc_luna_qualification_20260909')
LEDGER = Path('results/nonmyopic/openrouter_daily_budget/2026-09-09.json')
PROTOCOL = Path('results/nonmyopic/REARC_LUNA_QUALIFICATION_PROTOCOL_20260909.md')
PROTOCOL_SHA = '009912ad609ee29f43d4cfc3143c1882a344c36e9cfde886644ddaf11c171136'
RESERVE = Decimal('.06')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_route(endpoint):
    prices = endpoint['pricing']
    ceilings = {'prompt': '.0000002', 'completion': '.0000012',
                'input_cache_write': '.00000025', 'input_cache_read': '.0000002'}
    for key, ceiling in ceilings.items():
        if money(prices[key]) > Decimal(ceiling):
            raise ValueError('price ceiling')
    # Search is dormant: request bodies cannot contain tools, plugins or search options.
    for key, value in prices.items():
        if key not in {*ceilings, 'web_search', 'discount', 'overrides'} and money(value):
            raise ValueError('unbudgeted fee')
    for override in prices.get('overrides', []):
        if type(override.get('min_prompt_tokens')) is not int or override['min_prompt_tokens'] <= 65536:
            raise ValueError('applicable price override')
    exposure = 65536 * (money(prices['prompt']) + money(prices['input_cache_write'])) + 16384 * money(prices['completion'])
    if exposure > RESERVE:
        raise ValueError('exposure exceeds reservation')


def validate_body(body):
    if set(body) != {'model','max_tokens','reasoning','seed','provider','messages','response_format'}:
        raise ValueError('request features changed')
    if (body['model'] != 'openai/gpt-5.6-luna' or body['max_tokens'] != 16384
            or body['reasoning'] != {'enabled':True,'effort':'medium','exclude':True}
            or body['provider'] != {'only':['openai'],'allow_fallbacks':False,'require_parameters':True,
                                   'max_price':{'prompt':.2,'completion':1.2}}
            or len(json.dumps(body).encode()) > 32768):
        raise ValueError('request cap or route changed')


def response_text(raw):
    u = raw['usage']
    r = u['completion_tokens_details']['reasoning_tokens']
    if (type(r) is not int or type(u['completion_tokens']) is not int
            or not 0 < r <= u['completion_tokens'] <= 16384
            or type(u['prompt_tokens']) is not int or not 0 <= u['prompt_tokens'] <= 65536
            or money(u['cost']) > RESERVE
            or raw['model'] != 'openai/gpt-5.6-luna' or raw['provider'] != 'OpenAI'):
        raise ValueError('receipt mismatch')
    choice, = raw['choices']
    if choice['finish_reason'] != 'stop' or not isinstance(choice['message']['content'], str):
        raise ValueError('incomplete response')
    return choice['message']['content']


class Probe(PaidProbe):
    def request(self, tag, body):
        validate_body(body)
        endpoint = route()
        validate_route(endpoint)
        save(self.root/(tag+'.route.json'), endpoint, exclusive=True)
        save(self.root/(tag+'.request.json'), body, exclusive=True)
        if self.accepted + self.reserve > self.cap or self.report['calls'] >= 12:
            raise RuntimeError('block cap')
        self.account()
        self.ledger['pending_reservations'] = dict(self.carry, **{self.root.name+':'+tag:float(self.reserve)})
        save(self.path, self.ledger)
        self.pending, self.dispatched = True, False
        self.account()
        self.report.update(current=tag, phase='http_attempt', calls=self.report['calls']+1)
        save(self.root/'result.json', self.report)
        self.dispatched = True
        raw = execute(body)
        save(self.root/(tag+'.response.json'), raw, exclusive=True)
        self.accepted += money(raw['usage']['cost'])
        self.ledger['recorded_actual_spend_usd'] = float(max(money(self.ledger['recorded_actual_spend_usd']), self.base+self.accepted))
        self.ledger['pending_reservations'] = dict(self.carry)
        save(self.path, self.ledger)
        self.pending = False
        save(self.root/'result.json', self.report)
        return response_text(raw)


class Bank:
    def __init__(self, root, source=None, replay=False):
        self.root, self.source, self.replay = root, source, replay
        self.cache = {}

    def saved(self, name, build):
        path = self.root/(name+'.json')
        if self.replay:
            return json.loads(path.read_text())
        value = build()
        save(path, value, exclusive=True)
        return value

    def evaluate(self, graph, inputs):
        outputs = []
        for x in inputs:
            key = 'execution_'+digest([graph,x])
            if key not in self.cache:
                self.cache[key] = self.saved(key, lambda: execute_graph(graph,x))
            row = self.cache[key]
            outputs.append(row['output'] if row['status']=='ok' else None)
        return outputs

    def symbolic(self, inputs, outputs):
        key = 'symbolic_'+digest([inputs,outputs])
        if key not in self.cache:
            self.cache[key] = self.saved(key, lambda: execute_symbolic(inputs,outputs))
        row = self.cache[key]
        if row.get('status') != 'ok':
            raise ValueError('symbolic runtime failure')
        return row['graphs']

    def seal(self, forecasts):
        banked = self.saved('forecasts', lambda: forecasts)
        seal = self.saved('forecast_seal', lambda: {'sha256':digest(forecasts)})
        if banked != forecasts or seal != {'sha256':digest(banked)}:
            raise ValueError('forecast seal mismatch')
        self.sealed = seal

    def targets(self, cases):
        if not hasattr(self, 'sealed'):
            raise ValueError('forecasts not sealed')
        forecasts = json.loads((self.root/'forecasts.json').read_text())
        if digest(forecasts) != self.sealed['sha256']:
            raise ValueError('forecast changed before targets')
        values = self.saved('targets', lambda: self.source.targets(cases))
        for case, rows in zip(cases, values):
            hashes = [hashlib.sha256(json.dumps(y,separators=(',',':')).encode()).hexdigest() for y in rows]
            if hashes != case['target_hashes']:
                raise ValueError('target hash mismatch')
        return values


def run():
    bindings = sorted(str(p) for p in Path('scripts').glob('rearc_*.py')) + [
        str(PROTOCOL), 'results/nonmyopic/REARC_SOURCE_SCOPE_20260909.json',
        'scripts/paid_program_probe.py', 'scripts/openrouter_daily_budget.py',
        'scripts/deepcoder_luna_medium_probe.py', 'scripts/deepcoder_luna_transition.py',
        'scripts/deepcoder_proposal_gate.py']
    validate_route(route())
    with Probe(ROOT, LEDGER, '.72', PROTOCOL, PROTOCOL_SHA, bindings) as probe:
        probe.reserve = RESERVE
        try:
            with Examples() as source:
                bank = Bank(ROOT, source)
                cases = bank.saved('public', source.public)
                def targets():
                    probe.report['endpoints_opened'] = True
                    save(ROOT/'result.json', probe.report)
                    return bank.targets(cases)
                result = collect(cases, source.dsl, probe.request, bank.evaluate, bank.symbolic,
                                 bank.seal, targets)
                probe.report.update(result)
        except Exception as exc:
            probe.report['panel_failure'] = {'type':type(exc).__name__, 'message':str(exc)[:200]}
            raise
        finally:
            probe.report['artifact_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in ROOT.glob('*.json') if p.name != 'result.json'}
    return probe.report


def replay(root):
    report = json.loads((root/'result.json').read_text())
    actual = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.glob('*.json') if p.name not in {'result.json','replay.json'}}
    if actual != report['artifact_sha256']:
        raise ValueError('artifact identity mismatch')
    for path, sha in report['implementation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('implementation changed')
    cases = json.loads((root/'public.json').read_text())
    from scripts.rearc_graph_runtime import DSL_SHA
    import subprocess
    from scripts.rearc_source_scope import SOURCE, COMMIT
    dsl = subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':dsl.py']).decode()
    if hashlib.sha256(dsl.encode()).hexdigest() != DSL_SHA:
        raise ValueError('DSL identity')
    bank = Bank(root, replay=True)
    calls, costs = [], []
    def request(tag, body):
        validate_body(body)
        if json.loads((root/(tag+'.request.json')).read_text()) != body:
            raise ValueError('request replay mismatch')
        validate_route(json.loads((root/(tag+'.route.json')).read_text()))
        raw = json.loads((root/(tag+'.response.json')).read_text())
        calls.append(tag)
        costs.append(money(raw['usage']['cost']))
        return response_text(raw)
    try:
        result = collect(cases, dsl, request, bank.evaluate, bank.symbolic, bank.seal, lambda:bank.targets(cases))
    except Exception as exc:
        failure = {'type':type(exc).__name__, 'message':str(exc)[:200]}
        if report['status'] != 'failed_closed' or failure != report.get('panel_failure'):
            raise
        if any((root/(p+'.json')).exists() for p in ('forecasts','forecast_seal','targets')):
            raise ValueError('failed prefix opened downstream artifacts')
    else:
        if any(report[k] != value for k,value in result.items()):
            raise ValueError('result replay mismatch')
    if len(calls) != report['calls'] or sum(costs,Decimal()) != money(report['accepted_cost_usd']):
        raise ValueError('receipt coverage mismatch')
    return {'status':'exact_replay', 'calls':len(calls), 'new_calls':0, 'depth_authorized':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.execute == args.replay:
        parser.error('choose exactly one mode')
    if args.replay:
        print(json.dumps(replay(ROOT)))
    else:
        with LEDGER.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(run()))
