#!/usr/bin/env python3
"""Pure codec for sharded Number Game semantic bitmask support."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from scripts.number_game_extension_native_semantic_codec import audit_response_format, canonical_json, description_lexically_valid, parse_audit


INTERFACE_VERSION = "number-game-sharded-bitmask-semantic-support-1"
HISTORIES = (((4, True),), ((4, True), (20, True)), ((3, False), (27, True)), ((14, True), (15, False)), ((2, False), (22, True)))
PROPOSAL_SEEDS = tuple(range(202608132800, 202608132830))
AUDIT_SEEDS = tuple(range(202608132900, 202608132910))
SHARDS_PER_DRAW = 3; ITEMS_PER_SHARD = 8; DRAWS = 10


def proposal_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "number_game_semantic_bitmask_shard", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["hypotheses"], "properties": {"hypotheses": {"type": "array", "minItems": 8, "maxItems": 8, "items": {"type": "object", "additionalProperties": False, "required": ["name", "description", "membership_mask"], "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 60}, "description": {"type": "string", "minLength": 1, "maxLength": 180}, "membership_mask": {"type": "string", "pattern": "^[01]{101}$"}}}}}}}}


def proposal_messages(history: Sequence[tuple[int, bool]], shard_index: int) -> list[dict[str, str]]:
    observations = [{"number": n, "answer": "YES" if y else "NO"} for n, y in history]
    system = "Propose exactly 8 distinct general semantic rules for subsets of integers 0 through 100. Return only required JSON. Each description states one coherent reusable ordinary-language rule without code, explicit member lists, lookup tables, exception lists, or mask references. membership_mask has exactly 101 bits; bit i is 1 exactly when i satisfies the rule. Obey all observations. The shard nonce requests diversity only."
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json({"observations": observations, "domain": "integers 0 through 100 inclusive", "diversity_shard_nonce": shard_index})}]


def parse_shard(raw: str, history: Sequence[tuple[int, bool]]) -> list[dict[str, Any]]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses"} or not isinstance(value["hypotheses"], list) or len(value["hypotheses"]) != 8: raise ValueError("shard shape changed")
    out=[]; names=set(); masks=set()
    for i,row in enumerate(value["hypotheses"]):
        if not isinstance(row,dict) or set(row)!={"name","description","membership_mask"}: raise ValueError(f"item {i} shape changed")
        name=" ".join(str(row["name"]).strip().split()); desc=" ".join(str(row["description"]).strip().split()); mask=row["membership_mask"]
        if not name or len(name)>60 or name.casefold() in names: raise ValueError(f"item {i} name invalid")
        if not description_lexically_valid(desc) or "mask" in desc.casefold(): raise ValueError(f"item {i} description invalid")
        if not isinstance(mask,str) or len(mask)!=101 or set(mask)-{"0","1"} or mask in {"0"*101,"1"*101} or mask in masks: raise ValueError(f"item {i} mask invalid")
        if any((mask[n]=="1") is not y for n,y in history): raise ValueError(f"item {i} contradicts history")
        names.add(name.casefold()); masks.add(mask); out.append({"name":name,"description":desc,"mask":mask,"mask_hash":hashlib.sha256(mask.encode()).hexdigest()})
    return out


def merge_draw(shards: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    if len(shards)!=3 or any(len(shard)!=8 for shard in shards): raise ValueError("draw shard coverage changed")
    rows=[row for shard in shards for row in shard]
    if len({row["mask"] for row in rows})!=24 or len({row["name"].casefold() for row in rows})!=24: raise ValueError("cross-shard duplicates")
    return rows


def probes_for(draw_index: int, hypothesis_index: int, history: Sequence[tuple[int,bool]]) -> tuple[int,...]:
    seed_tuple=PROPOSAL_SEEDS[3*draw_index:3*draw_index+3]; probes=[]
    for j in range(8):
        raw=f"{INTERFACE_VERSION}|{seed_tuple}|{hypothesis_index}|{j}".encode(); x=int.from_bytes(hashlib.sha256(raw).digest(),"big")%101
        while x in probes: x=(x+1)%101
        probes.append(x)
    observed=[n for n,_ in history]
    return tuple([x for x in probes if x not in observed][:8-len(observed)]+observed)


def audit_messages(rows: Sequence[dict[str,Any]], draw_index:int, history:Sequence[tuple[int,bool]]) -> list[dict[str,str]]:
    rules=[{"hypothesis_index":i,"name":row["name"],"description":row["description"],"probe_integers":list(probes_for(draw_index,i,history))} for i,row in enumerate(rows)]
    return [{"role":"system","content":"Independently interpret each ordinary-language subset rule. Return only required JSON. For every probe report true exactly when it satisfies the description. Use only name, description, and probes."},{"role":"user","content":canonical_json({"rules":rules})}]


def diagnostics(draws, audits):
    if len(draws)!=10 or len(audits)!=10: raise ValueError("ten draws required")
    total=0; valid=[]; draw_rows=[]
    for d,(rows,judgments) in enumerate(zip(draws,audits,strict=True)):
        scores=[]; good=set()
        for i,(row,answers) in enumerate(zip(rows,judgments,strict=True)):
            probes=probes_for(d,i,HISTORIES[d//2]); score=sum(answer is (row["mask"][p]=="1") for answer,p in zip(answers,probes,strict=True)); scores.append(score); total+=score
            if score>=7: good.add(row["mask_hash"])
        valid.append(good); draw_rows.append({"draw_index":d,"history_index":d//2,"unique_mask_count":len({r["mask_hash"] for r in rows}),"mask_hashes":[r["mask_hash"] for r in rows],"agreement_count":sum(scores),"semantic_valid_count":len(good),"agreement_histogram":{str(s):scores.count(s) for s in range(9)}})
    pools=[]
    for h in range(5):
        left={r["mask_hash"] for r in draws[2*h]}; right={r["mask_hash"] for r in draws[2*h+1]}
        pools.append({"history_index":h,"unique_mask_count":len(left|right),"second_draw_novel_count":len(right-left),"semantic_valid_unique_count":len(valid[2*h]|valid[2*h+1])})
    base=valid[0]|valid[1]; novel=[len((valid[2*h]|valid[2*h+1])-base) for h in range(1,5)]
    gates={"every_draw_exactly_24_valid_unique":all(r["unique_mask_count"]==24 for r in draw_rows),"every_pool_at_least_28_unique":all(r["unique_mask_count"]>=28 for r in pools),"every_second_draw_at_least_4_novel":all(r["second_draw_novel_count"]>=4 for r in pools),"pooled_agreement_at_least_90_percent":total>=1728,"every_draw_at_least_18_semantic_valid":all(r["semantic_valid_count"]>=18 for r in draw_rows),"every_pool_at_least_26_semantic_valid":all(r["semantic_valid_unique_count"]>=26 for r in pools),"every_later_history_at_least_12_novel_semantic_valid":all(x>=12 for x in novel),"observed_answer_obedience_exact":all(all((r["mask"][n]=="1") is y for n,y in HISTORIES[d//2]) for d,rows in enumerate(draws) for r in rows)}
    return {"total_judgments":1920,"agreement_count":total,"agreement_rate":total/1920,"draws":draw_rows,"pools":pools,"observed_history_novel_counts":novel},gates


__all__=["AUDIT_SEEDS","HISTORIES","PROPOSAL_SEEDS","audit_messages","audit_response_format","diagnostics","merge_draw","parse_audit","parse_shard","proposal_messages","proposal_response_format","probes_for"]
