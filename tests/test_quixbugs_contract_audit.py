import pytest
from scripts.quixbugs_contract_audit import analyze_driver, inventory


def test_driver_analysis_never_executes_source():
    text = '''raise RuntimeError("must not run")
module = __import__("correct_python_programs."+name)
print(py_testcase)
p = subprocess.Popen(["java", name], stdout=subprocess.PIPE)
'''
    report = analyze_driver(text)
    assert report['corrected_module_import'] and report['prints_complete_test_record']
    assert report['popen_calls_without_timeout'] == 1
    assert not report['source_code_executed_by_audit']
    clean = analyze_driver('x = 1')
    assert not clean['corrected_module_import'] and not clean['prints_complete_test_record']


def test_inventory_reads_paths_not_contents():
    tree = {'truncated': False, 'tree': [
        {'type': 'blob', 'path': 'python_programs/a.py'},
        {'type': 'blob', 'path': 'correct_python_programs/a.py'},
        {'type': 'blob', 'path': 'json_testcases/a.json'},
        {'type': 'tree', 'path': 'python_programs/b.py'},
        {'type': 'blob', 'path': 'elsewhere/a.py'}]}
    groups = inventory(tree)
    assert groups['python_programs'] == ['python_programs/a.py']
    assert groups['json_testcases'] == ['json_testcases/a.json']
    tree['truncated'] = True
    with pytest.raises(ValueError):
        inventory(tree)
