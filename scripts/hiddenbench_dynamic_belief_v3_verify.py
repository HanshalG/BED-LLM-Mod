#!/usr/bin/env python3
"""Independent no-producer-import replay for HiddenBench V3 serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


QUERY_IDS = ("Q1", "Q2", "Q3", "Q4")
CHANNEL_IDS = ("C1", "C2", "C3")
INTERFACE_VERSION = "hiddenbench-dynamic-belief-v3-serving-v1"
PROTOCOL_SHA256 = "cab055aed50b22abfc1b90e720523882d8d6320616f0ab5f474a9340bf428bba"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def probability_list(value: Any, *, id_key: str, expected_ids: Sequence[str]) -> list[float]:
    if not isinstance(value, list) or len(value) != len(expected_ids):
        raise RuntimeError("probability list length changed")
    found: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {id_key, "probability"}:
            raise RuntimeError("probability item shape changed")
        item_id = item[id_key]
        probability = item["probability"]
        if item_id not in expected_ids or item_id in found or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("probability item is invalid")
        found[item_id] = float(probability)
    if set(found) != set(expected_ids) or abs(sum(found.values()) - 1) > 1e-6:
        raise RuntimeError("probability list is not normalized")
    return [found[item_id] for item_id in expected_ids]


def parse_root(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"prior", "queries"}:
        raise RuntimeError("root shape changed")
    prior_items = value["prior"]
    if not isinstance(prior_items, list) or len(prior_items) not in (3, 4):
        raise RuntimeError("root option count changed")
    option_ids = tuple(f"O{index + 1}" for index in range(len(prior_items)))
    prior = probability_list(prior_items, id_key="option_id", expected_ids=option_ids)
    if not isinstance(value["queries"], list) or len(value["queries"]) != 4:
        raise RuntimeError("root query count changed")
    queries: dict[str, Any] = {}
    requests: set[str] = set()
    dimensions: set[str] = set()
    for query in value["queries"]:
        if not isinstance(query, dict) or set(query) != {"query_id", "request", "target_dimension", "channels", "likelihoods"}:
            raise RuntimeError("root query shape changed")
        query_id = query["query_id"]
        request = query["request"]
        dimension = query["target_dimension"]
        normalized_request = " ".join(str(request).strip().lower().split())
        normalized_dimension = " ".join(str(dimension).strip().lower().split())
        if query_id not in QUERY_IDS or query_id in queries or not normalized_request or not normalized_dimension or normalized_request in requests or normalized_dimension in dimensions or any(phrase in normalized_request for phrase in ("correct answer", "which option", "choose the answer")):
            raise RuntimeError("root query is invalid")
        requests.add(normalized_request)
        dimensions.add(normalized_dimension)
        if not isinstance(query["channels"], list) or len(query["channels"]) != 3:
            raise RuntimeError("channel count changed")
        channel_descriptions: dict[str, str] = {}
        normalized_channels: set[str] = set()
        for channel in query["channels"]:
            if not isinstance(channel, dict) or set(channel) != {"channel_id", "description"}:
                raise RuntimeError("channel shape changed")
            channel_id = channel["channel_id"]
            description = str(channel["description"]).strip()
            normalized = " ".join(description.lower().split())
            if channel_id not in CHANNEL_IDS or channel_id in channel_descriptions or not normalized or normalized in normalized_channels:
                raise RuntimeError("channel is invalid")
            channel_descriptions[channel_id] = description
            normalized_channels.add(normalized)
        likelihoods: dict[str, list[float]] = {}
        if not isinstance(query["likelihoods"], list) or len(query["likelihoods"]) != len(option_ids):
            raise RuntimeError("likelihood count changed")
        for row in query["likelihoods"]:
            if not isinstance(row, dict) or set(row) != {"option_id", "channels"} or row["option_id"] in likelihoods:
                raise RuntimeError("likelihood row changed")
            likelihoods[row["option_id"]] = probability_list(row["channels"], id_key="channel_id", expected_ids=CHANNEL_IDS)
        if set(likelihoods) != set(option_ids):
            raise RuntimeError("likelihood coverage changed")
        queries[query_id] = {"likelihoods": likelihoods, "channels": channel_descriptions}
    if set(queries) != set(QUERY_IDS):
        raise RuntimeError("query coverage changed")
    return {"option_ids": option_ids, "prior": prior, "queries": queries}


def parse_refresh(raw: str, option_ids: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"branches"} or not isinstance(value["branches"], list) or len(value["branches"]) != 12:
        raise RuntimeError("refresh shape changed")
    result = {query_id: {} for query_id in QUERY_IDS}
    for branch in value["branches"]:
        if not isinstance(branch, dict) or set(branch) != {"query_id", "channel_id", "belief"}:
            raise RuntimeError("refresh branch shape changed")
        query_id = branch["query_id"]
        channel_id = branch["channel_id"]
        if query_id not in QUERY_IDS or channel_id not in CHANNEL_IDS or channel_id in result[query_id]:
            raise RuntimeError("refresh branch coverage changed")
        result[query_id][channel_id] = probability_list(branch["belief"], id_key="option_id", expected_ids=option_ids)
    if any(set(result[query_id]) != set(CHANNEL_IDS) for query_id in QUERY_IDS):
        raise RuntimeError("refresh coverage incomplete")
    return result


def parse_routing(raw: str) -> dict[str, dict[str, dict[str, str]]]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"tasks"} or not isinstance(value["tasks"], list) or len(value["tasks"]) != 4:
        raise RuntimeError("routing shape changed")
    result = {}
    for task in value["tasks"]:
        if not isinstance(task, dict) or set(task) != {"slot", "mappings"} or task["slot"] in result or task["slot"] not in ("T1", "T2", "T3", "T4"):
            raise RuntimeError("routing task changed")
        mappings = {}
        if not isinstance(task["mappings"], list) or len(task["mappings"]) != 4:
            raise RuntimeError("routing mapping count changed")
        for mapping in task["mappings"]:
            if not isinstance(mapping, dict) or set(mapping) != {"query_id", "fact_id", "channel_id"} or mapping["query_id"] in mappings or mapping["query_id"] not in QUERY_IDS or mapping["fact_id"] not in ("F1", "F2", "F3", "F4") or mapping["channel_id"] not in CHANNEL_IDS:
                raise RuntimeError("routing mapping changed")
            mappings[mapping["query_id"]] = {"fact_id": mapping["fact_id"], "channel_id": mapping["channel_id"]}
        result[task["slot"]] = mappings
    if set(result) != {"T1", "T2", "T3", "T4"}:
        raise RuntimeError("routing task coverage changed")
    return result


def entropy(p): return -sum(x * math.log(x) for x in p if x > 0)
def tv(a, b): return 0.5 * sum(abs(x-y) for x, y in zip(a, b, strict=True))
def predictive(b, m): return [sum(b[o]*m[o][c] for o in range(len(b))) for c in range(3)]
def update(b, m, c):
    joint=[b[o]*m[o][c] for o in range(len(b))]; z=sum(joint)
    if z <= 0: raise RuntimeError("zero predictive branch")
    return [x/z for x in joint]
def eig(b, m):
    return entropy(b)-sum(p*entropy(update(b,m,c)) for c,p in enumerate(predictive(b,m)) if p>0)


def dynamic(prior, matrices, refreshed):
    root_h=entropy(prior); scores={}; second={}
    for q in QUERY_IDS:
        terminal=0.0; second[q]={}
        for ci,c in enumerate(CHANNEL_IDS):
            branch=refreshed[q][c]
            candidates={s:eig(branch,matrices[s]) for s in QUERY_IDS if s!=q}
            sq=sorted(candidates,key=lambda x:(-candidates[x],x))[0]
            second[q][c]=sq
            terminal += predictive(prior,matrices[q])[ci]*sum(p*entropy(update(branch,matrices[sq],sc)) for sc,p in enumerate(predictive(branch,matrices[sq])) if p>0)
        scores[q]=root_h-terminal
    order=sorted(QUERY_IDS,key=lambda q:(-scores[q],q))
    return {"scores":scores,"first_query_id":order[0],"margin":scores[order[0]]-scores[order[1]],"second_actions":second,"response_contingent_first_queries":sum(len(set(v.values()))>=2 for v in second.values())}


def task_summary(slot, root, refreshed):
    option_ids=root["option_ids"]
    matrices={q:[root["queries"][q]["likelihoods"][o] for o in option_ids] for q in QUERY_IDS}
    one={q:eig(root["prior"],matrices[q]) for q in QUERY_IDS}; order=sorted(QUERY_IDS,key=lambda q:(-one[q],q))
    option_sensitive=sum(max(tv(matrices[q][i],matrices[q][j]) for i in range(len(option_ids)) for j in range(i+1,len(option_ids)))>=.10 for q in QUERY_IDS)
    increases=[]; exact_dist=[]; within=[]
    for q in QUERY_IDS:
        for ci,c in enumerate(CHANNEL_IDS):
            branch=refreshed[q][c]
            increases.append(sum(branch[o]*matrices[q][o][ci] for o in range(len(option_ids)))-sum(root["prior"][o]*matrices[q][o][ci] for o in range(len(option_ids))))
            exact_dist.append(tv(branch,update(root["prior"],matrices[q],ci)))
        within.extend(tv(refreshed[q][CHANNEL_IDS[i]],refreshed[q][CHANNEL_IDS[j]]) for i in range(3) for j in range(i+1,3))
    dyn=dynamic(root["prior"],matrices,refreshed)
    exact={q:{c:update(root["prior"],matrices[q],ci) for ci,c in enumerate(CHANNEL_IDS)} for q in QUERY_IDS}
    return {"slot":slot,"option_count":len(option_ids),"option_sensitive_queries":option_sensitive,"max_one_step_eig":one[order[0]],"one_step_eig_range":max(one.values())-min(one.values()),"myopic_first_query_id":order[0],"myopic_margin":one[order[0]]-one[order[1]],"dynamic":dyn,"fixed":dynamic(root["prior"],matrices,exact),"refresh":{"obedient_branch_count":sum(x>0 for x in increases),"mean_compatibility_increase":sum(increases)/12,"mean_exact_bayes_tv":sum(exact_dist)/12,"branches_exact_bayes_tv_at_least_001":sum(x>=.01 for x in exact_dist),"mean_within_query_pairwise_tv":sum(within)/12}}


def verify(run_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    raw_path=run_dir/"private/RAW_RESPONSES.json"; result_path=run_dir/"LABEL_FREE_RESULT.json"
    raw=load_object(raw_path); public=load_object(result_path)
    if set(raw)!={"roots","refreshes","router","auditor"} or {k:len(v) for k,v in raw.items()}!={"roots":4,"refreshes":4,"router":1,"auditor":1}: raise RuntimeError("raw bank incomplete")
    roots=[parse_root(x) for x in raw["roots"]]
    refreshes=[parse_refresh(x,r["option_ids"]) for x,r in zip(raw["refreshes"],roots,strict=True)]
    router=parse_routing(raw["router"][0]); auditor=parse_routing(raw["auditor"][0])
    tasks=[task_summary(f"T{i+1}",r,f) for i,(r,f) in enumerate(zip(roots,refreshes,strict=True))]
    semantic={"tasks":tasks,"changed_first_query_tasks":sum(t["dynamic"]["first_query_id"]!=t["myopic_first_query_id"] for t in tasks),"distinct_fact_counts":{slot:len({m["fact_id"] for m in mappings.values()}) for slot,mappings in router.items()},"exact_router_auditor_agreement":router==auditor}
    replay_gates={
        "option_sensitive_likelihoods":all(t["option_sensitive_queries"]>=2 for t in tasks),
        "nondegenerate_root_eig":all(t["max_one_step_eig"]>=.005 and t["one_step_eig_range"]>=.001 and t["myopic_margin"]>=.0001 for t in tasks),
        "refresh_answer_obedience":all(t["refresh"]["obedient_branch_count"]>=10 and t["refresh"]["mean_compatibility_increase"]>=.01 for t in tasks),
        "refresh_calibrated_irreducibility":all(.01<=t["refresh"]["mean_exact_bayes_tv"]<=.20 and t["refresh"]["branches_exact_bayes_tv_at_least_001"]>=6 and t["refresh"]["mean_within_query_pairwise_tv"]>=.10 for t in tasks),
        "dynamic_planning_nondegenerate":all(t["dynamic"]["margin"]>=.0001 and t["dynamic"]["response_contingent_first_queries"]>=2 for t in tasks),
        "dynamic_changes_first_query":semantic["changed_first_query_tasks"]>=2,
        "independent_routing_consensus":router==auditor and all(v>=3 for v in semantic["distinct_fact_counts"].values()),
    }
    gates={
        "exact_interface":public.get("interface_version")==INTERFACE_VERSION,
        "exact_protocol":public.get("protocol_sha256")==PROTOCOL_SHA256,
        "raw_hash_matches":public.get("raw_response_sha256")==file_digest(raw_path),
        "semantic_replays_exactly":public.get("semantic")==semantic,
        "routing_replays_exactly":public.get("consensus_routing")==router if router==auditor else public.get("consensus_routing")=={},
        "semantic_gates_replay_exactly":all(public.get("gates",{}).get(k) is v for k,v in replay_gates.items()),
        "transport_and_budget_gates_are_true":all(public.get("gates",{}).get(k) is True for k in ("exact_transport","strict_complete_parse","within_run_cap")),
        "status_and_authority_match":public.get("status")=="serving_pass" and public.get("authorizes")=="endpoint_only" and all(public.get("gates",{}).values()),
        "labels_remain_closed":public.get("registered_answers_opened") is False and public.get("endpoint_scores_opened") is False,
    }
    verification={"schema_version":1,"interface_version":"hiddenbench-dynamic-belief-v3-verifier-v1","status":"verification_pass" if all(gates.values()) else "verification_failed","gates":gates,"replayed_semantic":semantic,"replayed_semantic_gates":replay_gates,"registered_answers_loaded":False,"model_calls_made":0}
    if output_path is not None: output_path.write_text(json.dumps(verification,indent=2,sort_keys=True)+"\n")
    if verification["status"]!="verification_pass": raise RuntimeError("V3 independent verification failed")
    return verification


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--output",type=Path); args=parser.parse_args()
    print(json.dumps(verify(args.run_dir.resolve(),output_path=args.output),indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
