#!/usr/bin/env python3
"""Audit terminal closure of sharded Number Game mask generation."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from scripts.number_game_sharded_bitmask_semantic_codec import HISTORIES
BINDING="6543e6f50090bcdcb4bb7f597b3f7f28f5f67cb2114491588b7f30543aef0e10"; OPENING=220.134128880; EVENT=re.compile(r'^\{.*"event": "llm_token_usage".*\}$')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root:Path,*,output:Path|None=None):
    run=root/"mechanics-20260813"; raw=json.loads((run/"private/RAW_RESPONSES.json").read_text()); failure=json.loads((root/"DAILY_FAILURE_20260813.json").read_text()); ledger_path=root.parent/"openrouter_daily_budget/2026-08-13-number-game-sharded-bitmask-semantic.json"; ledger=json.loads(ledger_path.read_text()); events=[json.loads(x) for x in (run/"run.log").read_text().splitlines() if EVENT.match(x)]
    bad=0;items=0;bad_shards=0
    for index,response in enumerate(raw.get("proposal",[])):
        rows=json.loads(response)["hypotheses"]; history=HISTORIES[(index//3)//2]; local=0
        for row in rows:
            mismatch=any((row["membership_mask"][n]=="1") is not y for n,y in history); bad+=mismatch;local+=mismatch;items+=1
        bad_shards+=local>0
    local_cost=sum(float(x.get("cost_usd",0)) for x in events); recorded=max(float(ledger["closing_total_usage_usd"])-OPENING,float(ledger["execution_opening_total_usage_usd"])-OPENING+local_cost)
    gates={"binding_and_ledger":digest(root/"EXECUTION_BINDING.json")==BINDING and failure.get("ledger_sha256")==digest(ledger_path),"exact_clean_transport":len(events)==30 and all(x.get("finish_reasons")==["stop"] and int(x.get("reasoning_tokens",-1))==0 for x in events),"all_json_parseable":len(raw.get("proposal",[]))==30 and raw.get("audit")==[],"systematic_history_failure":items==240 and bad==163 and bad_shards==30,"cost_reconciles":abs(local_cost-float(failure["actual_cost_usd"]))<1e-12 and abs(recorded-float(ledger["recorded_actual_spend_usd"]))<1e-12,"authority_closed":failure.get("status")=="failed_closed" and failure.get("authorizes")=="nothing" and failure.get("targets_opened") is False and failure.get("endpoints_opened") is False}
    result={"schema_version":1,"interface_version":"number-game-sharded-bitmask-terminal-audit-1","status":"terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed","decision":"close_explicit_mask_generation","authorizes":"fresh_factorized_design_only" if all(gates.values()) else "nothing","gates":gates,"mechanics":{"proposal_requests":30,"audit_requests":0,"hypotheses":items,"history_contradicting_hypotheses":bad,"history_contradicting_rate":bad/items,"history_contradicting_shards":bad_shards,"actual_cost_usd":local_cost},"targets_opened":False,"endpoints_opened":False,"model_calls_made":0}
    if output: output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if result["status"]!="terminal_audit_pass": raise RuntimeError("sharded terminal audit failed")
    return result
