#!/usr/bin/env python3
"""Produce sharded Number Game semantic bitmask support."""

from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
from typing import Any,Callable,Protocol,Sequence
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_bitmask_semantic_gate import SeededAdapter,build_adapter as _build_adapter
from scripts.number_game_sharded_bitmask_semantic_codec import AUDIT_SEEDS,HISTORIES,PROPOSAL_SEEDS,audit_messages,audit_response_format,diagnostics,merge_draw,parse_audit,parse_shard,proposal_messages,proposal_response_format

INTERFACE_VERSION="number-game-sharded-bitmask-semantic-support-1"; PROPOSAL_MODEL_ID="qwen/qwen3.7-plus"; AUDIT_MODEL_ID="openai/gpt-5.6-luna"; PROPOSAL_MAX_TOKENS=1300; AUDIT_MAX_TOKENS=1200; STAGE_CAP_USD=.08
PROTOCOL=REPO_ROOT/"results/nonmyopic/NUMBER_GAME_SHARDED_BITMASK_SEMANTIC_SUPPORT_GATE_PROTOCOL_20260813.md"; PROTOCOL_SHA256="6c229c85185d79d65b942482c93f448641ae89277b486ca27832c4300c4d2fcb"
class Adapter(Protocol):
    def complete(self,messages,seeds,*,response_format,max_tokens):...
    def usage_snapshot(self):...
    def records(self):...
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def build_adapter(**kwargs)->SeededAdapter: return _build_adapter(**kwargs)
def usage(adapter):
    v=adapter.usage_snapshot(); return {"accepted_requests":int(v.get("adapter_requests",0)),"http_attempts":int(v.get("http_attempts",0)),"retries":int(v.get("retry_count",0)),"reasoning_tokens":int(v.get("adapter_reasoning_tokens",0)),"forced_exits":int(v.get("forced_exits",0)),"forced_final_requests":int(v.get("forced_final_requests",0)),"cost_usd":float(v.get("adapter_cost_usd",0))}
def run_gate(*,output_dir:Path,proposal_adapter:Adapter,audit_factory:Callable[[Sequence[list[dict[str,str]]]],Adapter]):
    if digest(PROTOCOL)!=PROTOCOL_SHA256: raise RuntimeError("sharded protocol changed")
    output_dir.mkdir(parents=True,exist_ok=True); private=output_dir/"private"; private.mkdir(parents=True,exist_ok=True); raw_path=private/"RAW_RESPONSES.json"; raw={"proposal":[],"audit":[]}
    messages=[]
    for draw in range(10):
        for shard in range(3): messages.append(proposal_messages(HISTORIES[draw//2],shard))
    proposal_raw=proposal_adapter.complete(messages,PROPOSAL_SEEDS,response_format=proposal_response_format(),max_tokens=PROPOSAL_MAX_TOKENS); raw["proposal"]=proposal_raw; checkpoint(raw_path,raw)
    parsed=[parse_shard(response,HISTORIES[(index//3)//2]) for index,response in enumerate(proposal_raw)]
    draws=[merge_draw(parsed[3*d:3*d+3]) for d in range(10)]
    audit_batch=[audit_messages(draws[d],d,HISTORIES[d//2]) for d in range(10)]; audit_adapter=audit_factory(audit_batch)
    audit_raw=audit_adapter.complete(audit_batch,AUDIT_SEEDS,response_format=audit_response_format(),max_tokens=AUDIT_MAX_TOKENS); raw["audit"]=audit_raw; checkpoint(raw_path,raw)
    semantic,semantic_gates=diagnostics(draws,[parse_audit(x) for x in audit_raw]); adapters=(proposal_adapter,audit_adapter); phases=[usage(x) for x in adapters]; totals={k:sum(p[k] for p in phases) for k in phases[0]}; records=[r for a in adapters for r in a.records()]
    transport={"exact_40_accepted_and_http":totals["accepted_requests"]==totals["http_attempts"]==40,"zero_retries_reasoning_forced":totals["retries"]==totals["reasoning_tokens"]==totals["forced_exits"]==totals["forced_final_requests"]==0,"all_clean_stops":len(records)==40 and all(r["finish_reasons"]==["stop"] for r in records),"exact_models_and_seeds":{r["seed"] for r in records}==set(PROPOSAL_SEEDS+AUDIT_SEEDS) and all(r["model_requested"]==r["model_returned"]==(PROPOSAL_MODEL_ID if r["seed"] in PROPOSAL_SEEDS else AUDIT_MODEL_ID) for r in records),"within_stage_cap":totals["cost_usd"]<=STAGE_CAP_USD+1e-12}
    gates={**transport,**semantic_gates}; passed=all(gates.values()); result={"schema_version":1,"interface_version":INTERFACE_VERSION,"status":"mechanics_pass" if passed else "mechanics_failed_closed","decision":"freeze_fresh_scientific_protocol" if passed else "close_exact_sharded_interface","authorizes":"separate_protocol_only" if passed else "nothing","protocol_sha256":PROTOCOL_SHA256,"raw_response_sha256":digest(raw_path),"semantic":semantic,"transport":{"phases":phases,"totals":totals,"requests":records},"gates":gates,"targets_opened":False,"endpoints_opened":False}; checkpoint(output_dir/"LABEL_FREE_RESULT.json",result); return result
