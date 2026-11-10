"""Record actual gauge nodes on hosts lacking Unix's os.uname (Windows)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_field_gauge_node import _reservoir
from umwelt.substrate.field_gauge import record_field_node, field_lineage


class PortableHost(unittest.TestCase):
    def test_real_field_record_without_uname_preserves_lineage_and_empty_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = getattr(os, 'uname', None)
            if original is not None:
                del os.uname
            try:
                with patch('platform.node', return_value='lichen-laptop'):
                    first = record_field_node(_reservoir(), archive_root=root, git_commit=False)
                    self.assertEqual(first['host'], 'lichen-laptop')
                    self.assertTrue((root / first['stored']).exists())
                    self.assertIn('canon_hash_v2', first['params'])
                    self.assertIn('wilson_signs', first['invariants'])
                    self.assertIsNone(record_field_node(_reservoir(), archive_root=root, git_commit=False))
                    changed = record_field_node(_reservoir(j=.45), archive_root=root, git_commit=False)
                    self.assertEqual(changed['parents'], [first['sha256'][:8]])
                    self.assertEqual(len(field_lineage(root)), 2)
            finally:
                if original is not None:
                    os.uname = original


if __name__ == '__main__':
    unittest.main()
