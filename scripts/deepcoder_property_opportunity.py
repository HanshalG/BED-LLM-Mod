"""Frozen source-only finite-reference screen for one-bit experiment semantics."""
import hashlib
from pathlib import Path
from scripts.deepcoder_opportunity import load_dsl, sample_program, sample_input
from scripts.deepcoder_local_horizon_audit import measure
from scripts.deepcoder_proposal_gate import save
from environments.program_induction.local_support import evaluate
from environments.program_induction.property_observation import observe, PROPERTIES
from environments.program_induction.prediction import category, digest, canonical


def main():
    dsl, panels = load_dsl(), []
    protocol = Path('results/nonmyopic/DEEPCODER_PROPERTY_OPPORTUNITY_PROTOCOL_20260909.md')
    for k in range(4):
        programs = [sample_program(dsl,29100000+1000*k+i) for i in range(128)]
        xs = [sample_input(30100000+1000*k+j) for j in range(40)]
        matrix = []
        for p in programs:
            queries = [str(int(observe(evaluate(p,x),prop))) for x,prop in zip(xs[:8],PROPERTIES)]
            matrix.append(queries+[category(evaluate(p,x)) for x in xs[8:]])
        try:
            result = dict(status='complete', **measure(matrix))
        except (TimeoutError, RuntimeError) as exc:
            result = dict(status='incomplete', error_type=type(exc).__name__)
        panels.append(dict(index=k,result=result,query_inputs=xs[:8],target_inputs=xs[8:],
                           program_sha256=digest([str(p) for p in programs]),matrix_sha256=digest(matrix)))
    report = dict(panels=panels,protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                  model_calls=0,cost_usd=0,interpretation='finite_empirical_prior_reference_not_full_grammar',
                  llm_efficacy=False)
    if all(p['result']['status']=='complete' for p in panels):
        means = {key:sum(p['result']['terminal_risk'][key] for p in panels)/4
                 for key in panels[0]['result']['terminal_risk']}
        report['mean_terminal_risk'] = means
        report['successive_5pct_internal_gap'] = means['h2']<=.95*means['h1'] and means['h3']<=.95*means['h2']
    save(Path('results/nonmyopic/DEEPCODER_PROPERTY_OPPORTUNITY_20260909.json'),report,exclusive=True)
    print(canonical(report))


if __name__ == '__main__':main()
