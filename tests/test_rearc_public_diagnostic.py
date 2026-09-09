import importlib.util
from pathlib import Path
import sys
from scripts import rearc_graph_worker, rearc_program_graph


def worker(monkeypatch):
    monkeypatch.setitem(sys.modules,'rearc_graph_worker',rearc_graph_worker)
    monkeypatch.setitem(sys.modules,'rearc_program_graph',rearc_program_graph)
    spec=importlib.util.spec_from_file_location('diagnostic',Path('scripts/rearc_public_diagnostic_worker.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_error_pinpoint_and_no_grid_contents_in_trace(monkeypatch):
    m=worker(monkeypatch)
    graph={'steps':[{'id':'x0','op':'first','args':['I']},
                    {'id':'x1','op':'gridonly','args':['x0']}],'output':'x1'}
    def gridonly(x): return tuple(tuple(row) for row in x)
    result=m.diagnose(graph,[[1,2],[3,4]],{'first':lambda x:x[0],'gridonly':gridonly},
                      {'first','gridonly'},set())
    assert result['status']=='execution_error'
    assert result['trace'][0]['result']=={'kind':'tuple','length':2}
    assert result['trace'][1]['error_type']=='TypeError'
    assert 'output' not in result


def test_good_grid_and_object_output(monkeypatch):
    m=worker(monkeypatch)
    g={'steps':[{'id':'x0','op':'identity','args':['I']}],'output':'x0'}
    assert m.diagnose(g,[[1]],{'identity':lambda x:x},{'identity'},set())['status']=='ok'
    assert m.diagnose(g,[[1]],{'identity':lambda x:frozenset()},{'identity'},set())['status']=='invalid_output'
