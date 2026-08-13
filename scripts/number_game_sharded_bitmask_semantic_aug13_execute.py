#!/usr/bin/env python3
"""Execute the fresh Aug 13 sharded Number Game semantic gate once."""
from __future__ import annotations
import argparse,hashlib,json,math,os
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from scripts import number_game_sharded_bitmask_semantic_gate as serving
from scripts import number_game_sharded_bitmask_semantic_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_sharded_bitmask_semantic_codec import HISTORIES,proposal_messages
from scripts.number_game_bitmask_semantic_aug13_execute import read_catalog,validate_catalog,validate_live,prior_spend
from scripts.openrouter_daily_budget import read_live_credits

INTERFACE_VERSION="number-game-sharded-bitmask-semantic-aug13-execute-1"; DATE="2026-08-13"; TZ="Europe/London"; OPENING=220.134128880; DAILY=5.; CAP=.08
ROOT=REPO_ROOT/"results/nonmyopic/number_game_sharded_bitmask_semantic_gate"; RUN=ROOT/"mechanics-20260813"; BINDING=ROOT/"EXECUTION_BINDING.json"; RESULT=ROOT/"DAILY_RESULT_20260813.json"; FAILURE=ROOT/"DAILY_FAILURE_20260813.json"; LEDGER=REPO_ROOT/"results/nonmyopic/openrouter_daily_budget/2026-08-13-number-game-sharded-bitmask-semantic.json"; RAW=RUN/"private/RAW_RESPONSES.json"; LABEL=RUN/"LABEL_FREE_RESULT.json"; VERIFY=RUN/"VERIFICATION.json"
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def exposure(messages,max_tokens,prices):
    values=[len(json.dumps(m,sort_keys=True,separators=(",",":")).encode())*float(prices["prompt_price_usd_per_token"])+max_tokens*float(prices["completion_price_usd_per_token"]) for m in messages]
    if not values or any(not math.isfinite(x) or x<=0 for x in values): raise RuntimeError("exposure malformed")
    return {"request_count":len(values),"exact_phase_exposure_usd":sum(values),"maximum_request_exposure_usd":max(values),"tracker_reservation_usd":len(values)*max(values),"request_exposures_usd":values,"max_output_tokens_per_request":max_tokens}
def authorize(model,messages,max_tokens,accepted,live_reader,catalog_reader):
    prices=validate_catalog(catalog_reader(),model); x=exposure(messages,max_tokens,prices); live=validate_live(live_reader()); reserve=x["tracker_reservation_usd"]
    if accepted+reserve>CAP+1e-12: raise RuntimeError("stage cap unavailable")
    if prior_spend(live)+reserve>DAILY+1e-12 or live["balance_usd"]+1e-12<reserve: raise RuntimeError("daily allowance unavailable")
    return {"model":prices,"exposure":x,"live":live}
def validate_bindings():
    b=load(BINDING); expected={"protocol":serving.PROTOCOL,"codec":REPO_ROOT/"scripts/number_game_sharded_bitmask_semantic_codec.py","producer":Path(serving.__file__).resolve(),"verifier":Path(verifier.__file__).resolve(),"wrapper":Path(__file__).resolve(),"account_utils":REPO_ROOT/"scripts/number_game_bitmask_semantic_aug13_execute.py","tests":REPO_ROOT/"tests/test_number_game_sharded_bitmask_semantic_codec.py"}
    for name,path in expected.items():
        row=b.get(name) or {}
        if row.get("path")!=str(path.relative_to(REPO_ROOT)) or row.get("sha256")!=digest(path): raise RuntimeError(f"binding changed:{name}")
    required={"date":DATE,"opening_total_usage_usd":OPENING,"daily_cap_usd":DAILY,"stage_cap_usd":CAP,"proposal_requests":30,"audit_requests":10,"maximum_http_attempts":40,"maximum_retries":0,"targets_authorized":False,"endpoints_authorized":False}
    if any(b.get(k)!=v for k,v in required.items()): raise RuntimeError("binding metadata changed")
    return {"execution_binding_sha256":digest(BINDING)}
def pristine(p): return not p.exists() or (p.is_dir() and not any(p.iterdir()))
def proposal_messages_all(): return [proposal_messages(HISTORIES[d//2],s) for d in range(10) for s in range(3)]
def preflight(*,now=None,live_reader=read_live_credits,catalog_reader=read_catalog):
    local=now.astimezone(ZoneInfo(TZ)) if now else datetime.now(ZoneInfo(TZ))
    if local.date().isoformat()!=DATE: raise RuntimeError("wrong date")
    bindings=validate_bindings()
    if RESULT.exists() or FAILURE.exists() or not pristine(RUN) or not pristine(LEDGER): raise RuntimeError("path not pristine")
    auth=authorize(serving.PROPOSAL_MODEL_ID,proposal_messages_all(),serving.PROPOSAL_MAX_TOKENS,0.,live_reader,catalog_reader)
    return {"schema_version":1,"interface_version":INTERFACE_VERSION,"status":"ready_without_paid_calls","bindings":bindings,"proposal_authorization":auth,"budget":{"opening_total_usage_usd":OPENING,"prior_account_spend_usd":prior_spend(auth["live"]),"daily_cap_usd":DAILY,"stage_cap_usd":CAP},"model_calls_made":0,"files_written":0}
def cost(a): return 0. if a is None else float(a.usage_snapshot().get("adapter_cost_usd",0))
def reconcile(ledger,status,local_cost,live):
    out=json.loads(json.dumps(ledger)); recorded=max(float(ledger["recorded_actual_spend_usd"]),prior_spend(live) if live else 0.,local_cost+float(ledger["execution_opening_total_usage_usd"])-OPENING)
    if recorded>DAILY+1e-12: raise RuntimeError("daily cap exceeded")
    out["recorded_actual_spend_usd"]=recorded; out["stage"].update({"status":status,"actual_cost_usd":local_cost})
    if live: out.update({"closing_total_credits_usd":live["total_credits_usd"],"closing_total_usage_usd":live["total_usage_usd"],"closing_balance_usd":live["balance_usd"]})
    return out
def execute(*,live_reader=read_live_credits,catalog_reader=read_catalog):
    ready=preflight(live_reader=live_reader,catalog_reader=catalog_reader); live=validate_live(live_reader()); pa=ready["proposal_authorization"]; ledger={"schema_version":1,"interface_version":INTERFACE_VERSION,"date":DATE,"timezone":TZ,"opening_total_usage_usd":OPENING,"execution_opening_total_credits_usd":live["total_credits_usd"],"execution_opening_total_usage_usd":live["total_usage_usd"],"execution_opening_balance_usd":live["balance_usd"],"recorded_actual_spend_usd":prior_spend(live),"daily_cap_usd":DAILY,"execution_binding_sha256":ready["bindings"]["execution_binding_sha256"],"proposal_authorization":pa,"stage":{"status":"authorized_pending","maximum_cost_usd":CAP,"maximum_http_attempts":40,"maximum_retries":0}}; checkpoint(LEDGER,ledger); p=None;a=None
    try:
        pcap=float(pa["exposure"]["maximum_request_exposure_usd"])
        def request_auth(cap):
            current=validate_live(live_reader())
            if prior_spend(current)+cap>DAILY+1e-12 or current["balance_usd"]+1e-12<cap: raise RuntimeError("request allowance lost")
        p=serving.build_adapter(model=serving.PROPOSAL_MODEL_ID,run_id="number-game-sharded-bitmask-semantic-20260813",output_dir=RUN,phase_exposure=float(pa["exposure"]["tracker_reservation_usd"]),request_cap=pcap,max_tokens=serving.PROPOSAL_MAX_TOKENS,authorize=lambda:request_auth(pcap))
        def factory(messages):
            nonlocal a,ledger
            auth=authorize(serving.AUDIT_MODEL_ID,messages,serving.AUDIT_MAX_TOKENS,cost(p),live_reader,catalog_reader); ledger["audit_authorization"]=auth; checkpoint(LEDGER,ledger); cap=float(auth["exposure"]["maximum_request_exposure_usd"]); a=serving.build_adapter(model=serving.AUDIT_MODEL_ID,run_id="number-game-sharded-bitmask-semantic-20260813",output_dir=RUN,phase_exposure=float(auth["exposure"]["tracker_reservation_usd"]),request_cap=cap,max_tokens=serving.AUDIT_MAX_TOKENS,authorize=lambda:request_auth(cap)); return a
        label=serving.run_gate(output_dir=RUN,proposal_adapter=p,audit_factory=factory); verifier.verify(RUN,output=VERIFY)
        try: closing=validate_live(live_reader())
        except Exception: closing=None
        actual=cost(p)+cost(a); ledger=reconcile(ledger,label["status"],actual,closing); checkpoint(LEDGER,ledger); terminal={"schema_version":1,"interface_version":INTERFACE_VERSION,"status":label["status"],"decision":label["decision"],"authorizes":label["authorizes"],"label_sha256":digest(LABEL),"raw_sha256":digest(RAW),"verification_sha256":digest(VERIFY),"ledger_sha256":digest(LEDGER),"actual_cost_usd":actual,"targets_opened":False,"endpoints_opened":False}; checkpoint(RESULT,terminal); return terminal
    except Exception as exc:
        try: closing=validate_live(live_reader())
        except Exception: closing=None
        actual=cost(p)+cost(a); ledger=reconcile(ledger,"failed_closed",actual,closing); checkpoint(LEDGER,ledger); failure={"schema_version":1,"interface_version":INTERFACE_VERSION,"status":"failed_closed","authorizes":"nothing","error_type":type(exc).__name__,"error":str(exc),"actual_cost_usd":actual,"ledger_sha256":digest(LEDGER),"raw_exists":RAW.exists(),"label_exists":LABEL.exists(),"verification_exists":VERIFY.exists(),"targets_opened":False,"endpoints_opened":False}; checkpoint(FAILURE,failure); raise
def main():
    p=argparse.ArgumentParser();p.add_argument("--preflight",action="store_true");a=p.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"): raise RuntimeError("key required")
    result=preflight() if a.preflight else execute();print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__": raise SystemExit(main())
