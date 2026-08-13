from __future__ import annotations
import hashlib,json
import pytest
from scripts import number_game_sharded_bitmask_semantic_codec as c
from scripts import number_game_sharded_bitmask_semantic_gate as gate
from scripts import number_game_sharded_bitmask_semantic_verify as verify
from pathlib import Path

def mask(draw,shard,i):
    bits=["1" if hashlib.sha256(f"s|{draw}|{shard}|{i}|{n}".encode()).digest()[0]<128 else "0" for n in range(101)]
    for n,y in c.HISTORIES[draw//2]: bits[n]="1" if y else "0"
    return "".join(bits)
def raw(draw,shard): return json.dumps({"hypotheses":[{"name":f"R {draw} {shard} {i}","description":f"Semantic family {draw} shard {shard} variant {i}","membership_mask":mask(draw,shard,i)} for i in range(8)]})
def draws(): return [c.merge_draw([c.parse_shard(raw(d,s),c.HISTORIES[d//2]) for s in range(3)]) for d in range(10)]
def audits(ds): return [[tuple(row["mask"][p]=="1" for p in c.probes_for(d,i,c.HISTORIES[d//2])) for i,row in enumerate(rows)] for d,rows in enumerate(ds)]

def test_merge_and_semantic_pass():
    ds=draws(); summary,gates=c.diagnostics(ds,audits(ds)); assert summary["agreement_rate"]==1; assert all(gates.values())
def test_probe_contains_history():
    p=c.probes_for(2,5,c.HISTORIES[1]); assert len(p)==len(set(p))==8; assert p[-2:]==(4,20)
def test_cross_shard_duplicate_rejected():
    shards=[c.parse_shard(raw(0,s),c.HISTORIES[0]) for s in range(3)]; shards[1][0]=dict(shards[0][0])
    with pytest.raises(ValueError): c.merge_draw(shards)
def test_bad_shard_and_history_rejected():
    value=json.loads(raw(0,0)); value["hypotheses"].pop()
    with pytest.raises(ValueError): c.parse_shard(json.dumps(value),c.HISTORIES[0])

class Fake:
    def __init__(self,model,seeds,responses): self.model=model; self.seeds=tuple(seeds); self.responses=responses
    def complete(self,messages,seeds,*,response_format,max_tokens): assert len(messages)==len(seeds)==len(self.responses); assert tuple(seeds)==self.seeds; return self.responses
    def usage_snapshot(self): return {"adapter_requests":len(self.responses),"http_attempts":len(self.responses),"retry_count":0,"adapter_reasoning_tokens":0,"forced_exits":0,"forced_final_requests":0,"adapter_cost_usd":.001}
    def records(self): return [{"seed":s,"model_requested":self.model,"model_returned":self.model,"finish_reasons":["stop"]} for s in self.seeds]
def audit_raw(ds):
    values=audits(ds); return [json.dumps({"judgments":[{"hypothesis_index":i,"memberships":list(row)} for i,row in enumerate(draw)]}) for draw in values]
def test_full_sharded_producer_replay(tmp_path:Path,monkeypatch):
    protocol=tmp_path/"p"; protocol.write_text("x"); monkeypatch.setattr(gate,"PROTOCOL",protocol); monkeypatch.setattr(gate,"PROTOCOL_SHA256",gate.digest(protocol))
    proposal_responses=[raw(d,s) for d in range(10) for s in range(3)]; ds=draws(); p=Fake(gate.PROPOSAL_MODEL_ID,c.PROPOSAL_SEEDS,proposal_responses); a=Fake(gate.AUDIT_MODEL_ID,c.AUDIT_SEEDS,audit_raw(ds))
    result=gate.run_gate(output_dir=tmp_path/"run",proposal_adapter=p,audit_factory=lambda _:a); assert result["status"]=="mechanics_pass"; assert len(result["semantic"]["draws"])==10; assert verify.verify(tmp_path/"run")["status"]=="verification_pass"
    value=json.loads(raw(0,0)); bits=list(value["hypotheses"][0]["membership_mask"]); bits[4]="0"; value["hypotheses"][0]["membership_mask"]="".join(bits)
    with pytest.raises(ValueError): c.parse_shard(json.dumps(value),c.HISTORIES[0])
