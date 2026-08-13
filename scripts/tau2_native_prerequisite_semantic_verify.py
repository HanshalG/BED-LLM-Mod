#!/usr/bin/env python3
"""Independently replay Tau2 native-prerequisite semantic mechanics artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_native_prerequisite_opportunity as exact
from scripts import tau2_native_prerequisite_source_manifest as source


SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-native-prerequisite-semantic-verification-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
MODEL_SEEDS = tuple(range(202608130100, 202608130106))
CONFIDENCE_MIN = 0.50
CONFIDENCE_MAX = 0.95
MMS_ROOTS = exact.MMS_ROOT_ACTIONS
MOBILE_ROOTS = exact.MOBILE_ROOT_ACTIONS
MMS_ACTION_FIELDS = {
    "apn_settings": {"mmsc_configured": [False, True]},
    "installed_apps": {"messaging_installed": [False, True]},
    "mms_probe": {"can_send": [False, True]},
    "network_mode": {"mode": ["2g_only", "4g_5g_preferred", "other"]},
    "network_status": {
        "airplane_mode": [False, True], "sim_active": [False, True],
        "connection": ["no_service", "connected", "other"],
        "signal": ["none", "poor", "excellent", "other"],
        "network_type": ["none", "2g", "5g", "other"],
        "mobile_data": [False, True], "data_roaming": [False, True],
        "wifi_radio": [False, True], "wifi_connected": [False, True],
    },
    "speed_test": {"speed": ["no_connection", "poor", "fair", "good", "excellent"]},
    "status_bar": {
        "airplane_mode": [False, True],
        "signal": ["none", "poor", "excellent", "other"],
        "network_type": ["none", "2g", "5g", "other"],
        "data_enabled": [False, True],
    },
    "wifi_calling": {"enabled": [False, True]},
    "messaging_permissions": {"sms": [False, True], "storage": [False, True], "phone": [False, True]},
}
MOBILE_ACTION_FIELDS = {
    "customer_lookup": {"account_active": [False, True]},
    "network_status": MMS_ACTION_FIELDS["network_status"],
    "payment_request": {"requested": [False, True]},
    "sim_status": {"active": [False, True]},
    "speed_test": MMS_ACTION_FIELDS["speed_test"],
    "status_bar": MMS_ACTION_FIELDS["status_bar"],
    "customer_bills": {"has_overage_charge": [False, True]},
    "data_usage": {"exhausted": [False, True]},
    "line_details": {"roaming_enabled": [False, True], "data_exhausted": [False, True], "line_active": [False, True]},
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def actions(family: str) -> dict[str, dict[str, Any]]:
    return MMS_ACTION_FIELDS if family.startswith("mms_") else MOBILE_ACTION_FIELDS


def selected_mechanics() -> list[dict[str, Any]]:
    tasks = json.loads(source.TASKS_PATH.read_text())
    splits = json.loads(source.SPLITS_PATH.read_text())
    selected, _ = source.select_episodes(
        [str(row["id"]) for row in tasks], list(splits["base"])
    )
    return list(selected["mechanics"])


def parse_raw(raw: str, family: str, count: int) -> list[dict[str, Any]]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"worlds"}:
        raise ValueError("raw semantic root changed")
    rows = payload["worlds"]
    fields = actions(family)
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("raw semantic world count changed")
    parsed = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"world_id", "predictions"} or row["world_id"] != f"w{index}":
            raise ValueError("raw semantic world identity changed")
        predictions = row["predictions"]
        if not isinstance(predictions, dict) or list(predictions) != list(fields):
            raise ValueError("raw semantic action order changed")
        clean = {}
        for action, field_values in fields.items():
            prediction = predictions[action]
            if not isinstance(prediction, dict) or set(prediction) != {*field_values, "confidence"}:
                raise ValueError("raw semantic fields changed")
            signature = {}
            for name, allowed in field_values.items():
                value = prediction[name]
                if value not in allowed or type(value) is not type(allowed[0]):
                    raise ValueError("raw semantic typed value invalid")
                signature[name] = value
            confidence = prediction["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not CONFIDENCE_MIN <= float(confidence) <= CONFIDENCE_MAX:
                raise ValueError("raw semantic confidence invalid")
            clean[action] = {"signature": signature, "confidence": float(confidence)}
        parsed.append({"world_id": f"w{index}", "predictions": clean})
    return parsed


def signature(value: Mapping[str, Any]) -> str:
    return canonical_json(dict(value))


def tables(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for action in rows[0]["predictions"]:
        predicted = [signature(row["predictions"][action]["signature"]) for row in rows]
        categories = sorted(set(predicted)) + ["OTHER"]
        action_rows = {}
        for row, own in zip(rows, predicted, strict=True):
            confidence = row["predictions"][action]["confidence"]
            rest = (1.0 - confidence) / (len(categories) - 1)
            distribution = {category: confidence if category == own else rest for category in categories}
            if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12):
                raise ValueError("independent likelihood row does not normalize")
            action_rows[row["world_id"]] = distribution
        result[action] = action_rows
    return result


def entropy(values: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0.0)


def posterior(prior, table, outcome):
    values = [prior[index] * table[world][outcome] for index, world in enumerate(table)]
    total = sum(values)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("independent posterior invalid")
    return [value / total for value in values]


def information(prior, table):
    start = entropy(prior)
    expected = 0.0
    for outcome in next(iter(table.values())):
        probability = sum(prior[index] * table[world][outcome] for index, world in enumerate(table))
        if probability:
            expected += probability * entropy(posterior(prior, table, outcome))
    return start - expected


def roots(family: str):
    return MMS_ROOTS if family.startswith("mms_") else MOBILE_ROOTS


def legal(family: str, root: str):
    base = [item for item in roots(family) if item != root]
    if family.startswith("mms_") and root == "installed_apps":
        base.append("messaging_permissions")
    if family == "mobile_abroad" and root == "customer_lookup":
        base.extend(("customer_bills", "data_usage", "line_details"))
    return tuple(sorted(base))


def depth_two(family: str, all_tables, root: str):
    first = all_tables[root]
    count = len(first)
    prior = [1.0 / count] * count
    value = information(prior, first)
    for outcome in next(iter(first.values())):
        probability = sum(prior[index] * first[world][outcome] for index, world in enumerate(first))
        if probability:
            branch = posterior(prior, first, outcome)
            value += probability * max(information(branch, all_tables[item]) for item in legal(family, root))
    return value


def parse_network(text: str):
    folded = text.casefold()
    def enum(options):
        return next((item for item in options if item in folded), "other")
    return {
        "airplane_mode": "airplane mode: on" in folded,
        "sim_active": "sim card status: active" in folded,
        "connection": enum(("no_service", "connected")),
        "signal": enum(("none", "poor", "excellent")),
        "network_type": enum(("none", "2g", "5g")),
        "mobile_data": "mobile data enabled: yes" in folded,
        "data_roaming": "data roaming enabled: yes" in folded,
        "wifi_radio": "wi-fi radio: on" in folded,
        "wifi_connected": "wi-fi connected: yes" in folded,
    }


def parse_speed(text: str):
    folded = text.casefold()
    if "no connection" in folded or "failed" in folded:
        return "no_connection"
    for value in ("poor", "fair", "good", "excellent"):
        if value in folded:
            return value
    raise ValueError("unknown speed output")


def parse_bar(text: str):
    folded = text.casefold()
    airplane = "airplane mode" in folded
    return {
        "airplane_mode": airplane,
        "signal": "none" if airplane else next((x for x in ("poor", "excellent") if x in folded), "other"),
        "network_type": "none" if airplane else next((x for x in ("2g", "5g") if x in folded), "other"),
        "data_enabled": "data enabled" in folded,
    }


def official_episode(episode: Mapping[str, Any], task_bank, get_environment):
    rows = []
    for world in episode["worlds"]:
        env = exact.initialized_environment(task_bank[world["task_id"]], get_environment)
        if episode["family"].startswith("mms_"):
            apn = str(env.user_tools.check_apn_settings()); apps = str(env.user_tools.check_installed_apps()); probe = str(env.user_tools.can_send_mms()); mode = str(env.user_tools.check_network_mode_preference()).casefold(); network = str(env.user_tools.check_network_status()); speed = str(env.user_tools.run_speed_test()); bar = str(env.user_tools.check_status_bar()); wifi = str(env.user_tools.check_wifi_calling_status()); permissions = str(env.user_tools.check_app_permissions("messaging")).casefold()
            rows.append({
                "apn_settings": {"mmsc_configured": "not set" not in apn.casefold()},
                "installed_apps": {"messaging_installed": "messaging" in apps.casefold()},
                "mms_probe": {"can_send": "cannot send" not in probe.casefold()},
                "network_mode": {"mode": "2g_only" if "2g_only" in mode else "4g_5g_preferred" if "4g_5g_preferred" in mode else "other"},
                "network_status": parse_network(network), "speed_test": {"speed": parse_speed(speed)}, "status_bar": parse_bar(bar),
                "wifi_calling": {"enabled": "turned on" in wifi.casefold()},
                "messaging_permissions": {"sms": "sms" in permissions, "storage": "storage" in permissions, "phone": "phone" in permissions},
            })
        else:
            phone = str(env.user_tools.surroundings.phone_number); customer = env.tools.get_customer_by_phone(phone); line = env.tools._get_line_by_phone(phone); cid, lid = str(customer.customer_id), str(line.line_id)
            network = str(env.user_tools.check_network_status()); payment = str(env.user_tools.check_payment_request()); sim = str(env.user_tools.check_sim_status()); speed = str(env.user_tools.run_speed_test()); bar = str(env.user_tools.check_status_bar()); bills = exact.plain_value(env.tools.get_bills_for_customer(cid)); usage = exact.plain_value(env.tools.get_data_usage(cid, lid)); details = exact.plain_value(env.tools.get_details_by_id(lid))
            rows.append({
                "customer_lookup": {"account_active": str(customer.account_status).casefold() == "active"}, "network_status": parse_network(network),
                "payment_request": {"requested": "no payment request" not in payment.casefold()}, "sim_status": {"active": "active and working" in sim.casefold()},
                "speed_test": {"speed": parse_speed(speed)}, "status_bar": parse_bar(bar),
                "customer_bills": {"has_overage_charge": any(str(item.get("item_type", "")).casefold() == "overage" for bill in bills for item in bill.get("line_items", []))},
                "data_usage": {"exhausted": float(usage["data_used_gb"]) > float(usage["data_limit_gb"])},
                "line_details": {"roaming_enabled": bool(details["roaming_enabled"]), "data_exhausted": float(details["data_used_gb"]) > 15.0, "line_active": str(details["status"]).casefold() == "active"},
            })
    return rows


def ranks(values):
    result = [0.0] * len(values); ordered = sorted(range(len(values)), key=lambda i: values[i]); start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and abs(values[ordered[end]] - values[ordered[start]]) <= 1e-12: end += 1
        for index in ordered[start:end]: result[index] = (start + end - 1) / 2
        start = end
    return result


def spearman(left, right):
    a, b = ranks(left), ranks(right); ma, mb = sum(a)/len(a), sum(b)/len(b)
    numerator = sum((x-ma)*(y-mb) for x,y in zip(a,b,strict=True)); denominator = math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    value=numerator/denominator if denominator else 0.0
    return round(max(-1.0,min(1.0,value)),15)


def replay(raw_bank: Mapping[str, Any]) -> dict[str, Any]:
    if raw_bank.get("model_id") != MODEL_ID or tuple(raw_bank.get("seeds", ())) != MODEL_SEEDS or len(raw_bank.get("responses", ())) != 6:
        raise ValueError("raw bank identity changed")
    episodes = selected_mechanics()
    parsed = [parse_raw(raw, ep["family"], len(ep["worlds"])) for raw, ep in zip(raw_bank["responses"], episodes, strict=True)]
    task_bank, get_environment = exact.initialize_tau2()
    actual = [official_episode(ep, task_bank, get_environment) for ep in episodes]
    cells = correct = native_exact = native_top = native_count = 0; total_brier = native_mass = native_brier = 0.0; family_brier = defaultdict(list); tvs=[]; sem_values=[]; src_values=[]; episode_metrics=[]
    for episode, rows, actual_rows in zip(episodes, parsed, actual, strict=True):
        family=episode["family"]; all_tables=tables(rows); root_values={a:information([1/len(rows)]*len(rows),all_tables[a]) for a in roots(family)}; depth_values={a:depth_two(family,all_tables,a) for a in roots(family)}; greedy=max(roots(family),key=lambda a:(root_values[a],a)); planned=max(roots(family),key=lambda a:(depth_values[a],a)); native="messaging_permissions" if family.startswith("mms_") else "line_details"
        for wi, observations in enumerate(actual_rows):
            for action, truth in observations.items():
                predicted=rows[wi]["predictions"][action]["signature"]; correct += signature(predicted)==signature(truth); cells += 1; category=signature(truth); distribution=all_tables[action][f"w{wi}"]; outcome=category if category in distribution else "OTHER"; score=sum((p-float(k==outcome))**2 for k,p in distribution.items()); total_brier += score; family_brier[family].append(score)
            truth=observations[native]; predicted=rows[wi]["predictions"][native]["signature"]; native_exact += signature(predicted)==signature(truth); distribution=all_tables[native][f"w{wi}"]; outcome=signature(truth) if signature(truth) in distribution else "OTHER"; post=posterior([1/len(rows)]*len(rows),all_tables[native],outcome); native_mass += post[wi]; native_top += post[wi] >= max(post)-1e-12; native_brier += sum((p-float(i==wi))**2 for i,p in enumerate(post)); native_count += 1
        for action in actions(family):
            table=all_tables[action]
            for i in range(len(rows)):
                for j in range(i+1,len(rows)):
                    if signature(actual_rows[i][action])==signature(actual_rows[j][action]): tvs.append(.5*sum(abs(table[f"w{i}"][k]-table[f"w{j}"][k]) for k in table[f"w{i}"]))
        canonical=[{a:signature(v) for a,v in row.items()} for row in actual_rows]; exact_values={a:exact.two_step_information(family,canonical,a) for a in roots(family)}
        for action in roots(family): sem_values.append(depth_values[action]); src_values.append(exact_values[action])
        episode_metrics.append({"family":family,"state_count":len(rows),"greedy_first_action":greedy,"depth_two_first_action":planned,"native_prerequisite":exact.prerequisite_action(family),"horizon_gain_nats":depth_values[planned]-depth_values[greedy],"root_information_nats":root_values,"two_step_information_nats":depth_values,"exact_two_step_information_nats":exact_values})
    family_means={key:sum(value)/len(value) for key,value in family_brier.items()}; metrics={"cell_count":cells,"typed_signature_accuracy":correct/cells,"mean_multiclass_brier":total_brier/cells,"family_mean_multiclass_brier":family_means,"native_followup_cell_count":native_count,"native_signature_exact_count":native_exact,"native_truth_top_rank_count":native_top,"native_mean_truth_posterior":native_mass/native_count,"native_mean_posterior_brier":native_brier/native_count,"equivalent_pair_count":len(tvs),"equivalent_mean_total_variation":sum(tvs)/len(tvs),"equivalent_max_total_variation":max(tvs),"semantic_source_two_step_spearman":spearman(sem_values,src_values),"mean_horizon_gain_nats":sum(row["horizon_gain_nats"] for row in episode_metrics)/6}
    gates={"exact_234_world_action_cells":cells==234,"typed_signature_accuracy_at_least_0_80":metrics["typed_signature_accuracy"]>=.8,"mean_multiclass_brier_at_most_0_20":metrics["mean_multiclass_brier"]<=.2,"each_family_brier_at_most_0_25":set(family_means)==set(source.EXPECTED_ELIGIBLE) and all(v<=.25 for v in family_means.values()),"exact_26_native_followup_cells":native_count==26,"native_signature_exact_at_least_24":native_exact>=24,"native_truth_top_rank_at_least_24":native_top>=24,"native_mean_truth_posterior_at_least_0_60":metrics["native_mean_truth_posterior"]>=.6,"native_mean_posterior_brier_at_most_0_20":metrics["native_mean_posterior_brier"]<=.2,"equivalent_mean_tv_at_most_0_05":metrics["equivalent_mean_total_variation"]<=.05,"equivalent_max_tv_at_most_0_15":metrics["equivalent_max_total_variation"]<=.15,"all_six_greedy_avoid_prerequisite":all(row["greedy_first_action"]!=row["native_prerequisite"] for row in episode_metrics),"all_six_depth_two_choose_prerequisite":all(row["depth_two_first_action"]==row["native_prerequisite"] for row in episode_metrics),"all_six_horizon_gain_at_least_0_05":all(row["horizon_gain_nats"]>=.05 for row in episode_metrics),"mean_horizon_gain_at_least_0_15":metrics["mean_horizon_gain_nats"]>=.15,"semantic_source_spearman_at_least_0_70":metrics["semantic_source_two_step_spearman"]>=.7}; gates["all_semantic_and_planning_gates_pass"]=all(gates.values())
    return {"metrics":metrics,"episode_metrics":episode_metrics,"gates":gates}


def verify(run_dir: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    result=load_object(run_dir/"RESULT.json"); raw=load_object(run_dir/"private/RAW_RESPONSES.json"); ordering=load_object(run_dir/"private/ORDERING.json"); privacy=load_object(run_dir/"private/PROMPT_PRIVACY.json")
    replayed=replay(raw)
    if result.get("metrics")!=replayed["metrics"] or result.get("episode_metrics")!=replayed["episode_metrics"] or result.get("gates")!=replayed["gates"]: raise ValueError("semantic mechanics replay differs")
    usage=result.get("usage",{}); serving=result.get("serving_gates",{}); expected_serving={"exact_six_accepted_requests":usage.get("adapter_requests")==6,"exact_six_http_attempts":usage.get("http_attempts")==6,"zero_retries":usage.get("retry_count")==0,"zero_reasoning_tokens":usage.get("adapter_reasoning_tokens")==0,"zero_forced_exits":usage.get("forced_exits")==0,"within_stage_cap":float(usage.get("run_cost_usd",math.inf))<=.1+1e-12}
    if serving!=expected_serving: raise ValueError("semantic serving gates differ")
    all_pass=all(expected_serving.values()) and replayed["gates"]["all_semantic_and_planning_gates_pass"]; expected_status="semantic_mechanics_pass" if all_pass else "semantic_mechanics_null"; expected_auth="prospective_development_protocol_only" if all_pass else "nothing"
    checks={"raw_replay_exact":True,"serving_replay_exact":True,"status_exact":result.get("status")==expected_status and result.get("authorizes")==expected_auth,"ordering_exact":ordering=={"all_raw_responses_banked":True,"official_mechanics_loaded_after_raw_bank":True} and result.get("ordering")==ordering,"privacy_exact":privacy.get("selected_task_ids_in_prompts") is False and privacy.get("source_fault_ids_in_prompts") is False and privacy.get("raw_tool_responses_in_prompts") is False and privacy.get("repair_or_endpoint_outcomes_in_prompts") is False,"downstream_unopened":result.get("development_confirmation_reserve_opened") is False and result.get("repair_or_task_success_endpoints_opened") is False}
    verification={"schema_version":SCHEMA_VERSION,"interface_version":INTERFACE_VERSION,"status":"verified" if all(checks.values()) else "invalid","expected_result_status":expected_status,"checks":checks,"all_pass":all(checks.values()),"result_sha256":sha256_file(run_dir/"RESULT.json"),"raw_bank_sha256":sha256_file(run_dir/"private/RAW_RESPONSES.json"),"model_calls_made":0,"cost_usd":0.0}
    if not verification["all_pass"]: raise ValueError("semantic mechanics verification failed")
    if output_path: output_path.write_text(json.dumps(verification,indent=2,sort_keys=True)+"\n")
    return verification


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("run_dir",type=Path); parser.add_argument("--output",type=Path); args=parser.parse_args(); print(json.dumps(verify(args.run_dir,output_path=args.output),indent=2,sort_keys=True))
