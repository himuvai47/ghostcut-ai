import unittest

from indexing_agent.config import IndexerConfig


class ConfigTests(unittest.TestCase):
    def test_signature_stable(self):
        a = IndexerConfig()
        b = IndexerConfig()
        self.assertEqual(a.index_signature(), b.index_signature())

    def test_signature_changes_with_semantic_settings(self):
        a = IndexerConfig()
        b = IndexerConfig(max_semantic_clip_sec=9.0)
        self.assertNotEqual(a.index_signature(), b.index_signature())
