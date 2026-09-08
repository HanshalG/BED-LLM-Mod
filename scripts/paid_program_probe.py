"""One-shot receipt and budget lifecycle shared by new program probes."""
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.deepcoder_proposal_gate import save, money
from scripts.deepcoder_luna_transition import execute
from scripts.deepcoder_luna_medium_probe import route
from scripts.openrouter_daily_budget import read_live_credits, budget_status, require_budget


class PaidProbe:
    def __init__(self, root, ledger_path, cap, protocol, protocol_sha, bindings):
        if root.exists():
            raise RuntimeError('already opened; no retry')
        if hashlib.sha256(protocol.read_bytes()).hexdigest() != protocol_sha:
            raise ValueError('protocol changed')
        self.root, self.path, self.cap = root, ledger_path, money(cap)
        self.reserve = Decimal('.04')
        self.ledger = json.loads(ledger_path.read_text())
        live = read_live_credits()
        s = require_budget(self.ledger, projected_cost_usd=float(self.cap), total_usage_usd=live['total_usage_usd'])
        if money(live['balance_usd']) < self.cap:
            raise RuntimeError('insufficient balance')
        self.base = money(s['spent_today_usd'])
        self.carry = dict(self.ledger.get('pending_reservations',{}))
        if sum((money(v) for v in self.carry.values()),Decimal()) > self.base:
            raise ValueError('unaccounted reservations')
        endpoint = route()
        root.mkdir()
        save(root/'route.json',endpoint,exclusive=True)
        self.accepted, self.uncertain = Decimal(), Decimal()
        self.pending = self.dispatched = False
        self.report = dict(status='incomplete',calls=0,endpoints_opened=False,depth_authorized=False,
            protocol_sha256=protocol_sha,
            implementation_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in bindings})
        self.ledger['recorded_actual_spend_usd'] = float(self.base)
        save(self.path,self.ledger)
        save(root/'result.json',self.report)

    def __enter__(self):
        return self

    def account(self):
        latest = json.loads(self.path.read_text())
        if latest.get('pending_reservations',{}) != self.ledger.get('pending_reservations',{}):
            raise RuntimeError('reservation conflict')
        self.ledger = latest
        live = read_live_credits()
        s = budget_status(self.ledger,total_usage_usd=live['total_usage_usd'])
        self.ledger['recorded_actual_spend_usd'] = float(max(money(s['spent_today_usd']),self.base+self.accepted))
        save(self.path,self.ledger)
        require_budget(self.ledger,projected_cost_usd=float(self.reserve),total_usage_usd=live['total_usage_usd'])
        if money(live['balance_usd']) < self.reserve:
            raise RuntimeError('insufficient dispatch balance')

    def request(self, tag, body):
        self.report.update(current=tag,phase='authorize')
        save(self.root/'result.json',self.report)
        save(self.root/(tag+'.route.json'),route(),exclusive=True)
        save(self.root/(tag+'.request.json'),body,exclusive=True)
        if self.accepted+self.reserve > self.cap:
            raise RuntimeError('block cap exceeded')
        self.account()
        self.ledger['pending_reservations'] = dict(self.carry,**{self.root.name+':'+tag:float(self.reserve)})
        save(self.path,self.ledger)
        self.pending,self.dispatched = True,False
        self.account()
        self.report.update(phase='http_attempt',calls=self.report['calls']+1)
        save(self.root/'result.json',self.report)
        self.dispatched = True
        raw = execute(body)
        save(self.root/(tag+'.response.json'),raw,exclusive=True)
        self.accepted += money(raw['usage']['cost'])
        self.ledger['recorded_actual_spend_usd'] = float(max(money(self.ledger['recorded_actual_spend_usd']),self.base+self.accepted))
        self.ledger['pending_reservations'] = dict(self.carry)
        save(self.path,self.ledger)
        self.pending = False
        self.report.update(phase='validate_response',accepted_cost_usd=float(self.accepted))
        save(self.root/'result.json',self.report)
        return raw

    def __exit__(self, typ, value, traceback):
        if typ is not None or self.report['status']=='incomplete':
            self.report.update(status='failed_closed',error_type=typ.__name__ if typ else 'UnfinishedRun')
        if self.pending and self.dispatched:
            self.uncertain = self.reserve
        self.ledger['pending_reservations'] = dict(self.carry,**(
            {self.root.name+':uncertain':float(self.uncertain)} if self.uncertain else {}))
        self.ledger['recorded_actual_spend_usd'] = float(max(money(self.ledger['recorded_actual_spend_usd']),
                                                          self.base+self.accepted+self.uncertain))
        save(self.path,self.ledger)
        self.report.update(accepted_cost_usd=float(self.accepted),uncertain_exposure_usd=float(self.uncertain),
            completed_at=datetime.now(ZoneInfo('Europe/London')).isoformat())
        save(self.root/'result.json',self.report)
        return isinstance(value,Exception)
