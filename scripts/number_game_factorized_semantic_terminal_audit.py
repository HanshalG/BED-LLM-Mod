#!/usr/bin/env python3
"""Audit terminal closure of factorized Number Game proposal mechanics."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from scripts.number_game_extension_native_semantic_codec import description_lexically_valid
BINDING="c3a40285a025dfdebb2f1891387b275164b838886d0768417405605b66efe9ff";OPENING=220.134128880
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root:Path,*,output:Path|None=None):
    run=root/"mechanics-20260813";raw=json.loads((run/"private/RAW_RESPONSES.json").read_text());failure=json.loads((root/"DAILY_FAILURE_20260813.json").read_text());ledger_path=root.parent/"openrouter_daily_budget/2026-08-13-number-game-factorized-semantic.json";ledger=json.loads(ledger_path.read_text());events=[json.loads(x) for x in (run/"run.log").read_text().splitlines() if '"event": "llm_token_usage"' in x]
    valid=[];invalid_lexical=0;invalid_observed=0
    for response in raw.get("proposal",[]):
        rows=json.loads(response)["hypotheses"];local=0
        for row in rows:
            desc=" ".join(str(row["description"]).strip().split());lex=description_lexically_valid(desc) and "mask" not in desc.casefold();obs=bool(re.search(r"\b(?:yes|no|observed|observation|answer)\b",desc,re.I));local+=lex and not obs;invalid_lexical+=not lex;invalid_observed+=obs
        valid.append(local)
    per_draw=[sum(valid[3*d:3*d+3]) for d in range(10)];local_cost=sum(float(x.get("cost_usd",0)) for x in events);recorded=max(float(ledger["closing_total_usage_usd"])-OPENING,float(ledger["execution_opening_total_usage_usd"])-OPENING+local_cost)
    gates={"binding_ledger":digest(root/"EXECUTION_BINDING.json")==BINDING and failure.get("ledger_sha256")==digest(ledger_path),"exact_clean_transport":len(events)==30 and all(x.get("finish_reasons")==["stop"] and int(x.get("reasoning_tokens",-1))==0 for x in events),"proposal_only":len(raw.get("proposal",[]))==30 and raw.get("translation")==[] and raw.get("audit")==[],"aggregate_signature":sum(valid)==231 and min(per_draw)==21 and invalid_lexical==2 and invalid_observed==7,"cost_reconciles":abs(local_cost-float(failure["actual_cost_usd"]))<1e-12 and abs(recorded-float(ledger["recorded_actual_spend_usd"]))<1e-12,"authority_closed":failure.get("status")=="failed_closed" and failure.get("authorizes")=="nothing" and failure.get("targets_opened") is False and failure.get("endpoints_opened") is False}
    result={"schema_version":1,"interface_version":"number-game-factorized-semantic-terminal-audit-1","status":"terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed","decision":"close_exact_factorized_interface","authorizes":"fresh_overgenerated_design_only" if all(gates.values()) else "nothing","gates":gates,"proposal":{"requests":30,"items":240,"valid_items":sum(valid),"invalid_lexical_items":invalid_lexical,"invalid_observed_language_items":invalid_observed,"per_draw_valid_counts":per_draw,"minimum_draw_valid":min(per_draw),"translation_requests":0,"audit_requests":0,"actual_cost_usd":local_cost},"targets_opened":False,"endpoints_opened":False,"model_calls_made":0}
    if output:output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if result["status"]!="terminal_audit_pass":raise RuntimeError("factorized terminal audit failed")
    return result
