#!/usr/bin/env python3
"""Independent replay of sharded semantic bitmask mechanics."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from scripts.number_game_sharded_bitmask_semantic_codec import HISTORIES,diagnostics,merge_draw,parse_audit,parse_shard
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    v=json.loads(p.read_text());
    if not isinstance(v,dict): raise RuntimeError("expected object")
    return v
def verify(run_dir:Path,*,output:Path|None=None):
    raw_path=run_dir/"private/RAW_RESPONSES.json"; raw=load(raw_path); public=load(run_dir/"LABEL_FREE_RESULT.json")
    if set(raw)!={"proposal","audit"} or len(raw["proposal"])!=30 or len(raw["audit"])!=10: raise RuntimeError("raw bank incomplete")
    shards=[parse_shard(x,HISTORIES[(i//3)//2]) for i,x in enumerate(raw["proposal"])]; draws=[merge_draw(shards[3*d:3*d+3]) for d in range(10)]; semantic,replay=diagnostics(draws,[parse_audit(x) for x in raw["audit"]]); pg=public.get("gates") or {}
    gates={"raw_hash":public.get("raw_response_sha256")==digest(raw_path),"semantic_replay":public.get("semantic")==semantic,"semantic_gates":all(pg.get(k) is v for k,v in replay.items()),"transport_present":all(k in pg for k in ("exact_40_accepted_and_http","zero_retries_reasoning_forced","all_clean_stops","exact_models_and_seeds","within_stage_cap")),"status_authority":(public.get("status")=="mechanics_pass") is all(bool(v) for v in pg.values()) and public.get("targets_opened") is False and public.get("endpoints_opened") is False}
    result={"schema_version":1,"interface_version":"number-game-sharded-bitmask-semantic-verifier-1","status":"verification_pass" if all(gates.values()) else "verification_failed","gates":gates,"model_calls_made":0,"targets_opened":False,"endpoints_opened":False}
    if output: output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if result["status"]!="verification_pass": raise RuntimeError("sharded verification failed")
    return result
