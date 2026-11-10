"""Supported consumer transport stays separate from declarative world data."""
import inspect
from pathlib import Path
import re
import subprocess
import sys

import pytest

from umweltd import UmweltClient
from umwelt.spec.declarative import audit_source


def test_client_alias_types_and_documented_methods():
    from umweltd.client import UmweltClient as original
    assert UmweltClient is original
    names = {'create_world', 'list_worlds', 'start_world', 'stop_world', 'ingest',
             'health', 'state', 'cognifold', 'belief', 'recommendations', 'bindings', 'snapshot', 'anchor'}
    assert {n for n in vars(UmweltClient) if not n.startswith('_')} == names
    document = (Path(__file__).parents[1]/'docs/DAEMON-API.md').read_text(encoding='utf-8')
    for name in [*names, '__init__']:
        method = getattr(UmweltClient, name)
        assert inspect.getdoc(method)
        signature = inspect.signature(method)
        assert signature.return_annotation is not inspect.Signature.empty
        assert all(p.annotation is not inspect.Parameter.empty
                   for n, p in signature.parameters.items() if n != 'self')
        if name != '__init__':
            assert f'`{name}(' in document
    assert (Path(__file__).parents[1]/'src/umweltd/py.typed').is_file()


def test_client_import_and_documentation_do_not_load_engine_or_open_network():
    doc = Path(__file__).parents[1]/'docs/DAEMON-API.md'
    examples = re.findall(r'^```python\n(.*?)^```', doc.read_text(encoding='utf-8'), re.S | re.M)
    assert examples
    code = "import sys\ndef audit(event, args):\n    if event in ('socket.connect', 'socket.getaddrinfo'):\n        raise AssertionError('unexpected connection')\n"
    code += "sys.addaudithook(audit)\n" + '\n'.join(examples)
    code += "\nassert not any(n.startswith(('numpy', 'umwelt.engine', 'umweltd.worker', 'umweltd.supervisor')) for n in sys.modules)\n"
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize('statement', [
    'from umweltd import UmweltClient',
    'from umweltd import UmweltClient as Client',
    'import umweltd as transport',
])
def test_transport_promotion_does_not_admit_network_client_to_declarative_specs(tmp_path, statement):
    source = tmp_path/'world.py'
    source.write_text(statement+'\n', encoding='utf-8')
    assert any('outside the declarative allowlist' in p for p in audit_source(source))


def test_public_registration_imports_still_pass_declarative_audit(tmp_path):
    source = tmp_path/'world.py'
    source.write_text('from umwelt import register_role_mode, register_observe_role, '
                      'register_role_emoji, register_node_icon\n', encoding='utf-8')
    assert audit_source(source) == []
