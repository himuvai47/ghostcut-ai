import unittest

from indexing_agent.config import IndexerConfig
from indexing_agent.ffmpeg_tools import scene_cuts_to_shots
from indexing_agent.models import TimeRange
from indexing_agent.sampling import representative_times
from indexing_agent.segmentation import select_change_boundaries


class SegmentationTests(unittest.TestCase):
    def test_scene_cut_conversion_covers_duration(self):
        shots = scene_cuts_to_shots([3.0, 3.3, 8.0], 10.0, 1.0)
        self.assertAlmostEqual(shots[0].start, 0.0)
        self.assertAlmostEqual(shots[-1].end, 10.0)
        for a, b in zip(shots, shots[1:]):
            self.assertAlmostEqual(a.end, b.start)

    def test_representative_times_inside_clip(self):
        cfg = IndexerConfig()
        seg = TimeRange(10.0, 16.0)
        times = representative_times(seg, cfg)
        self.assertGreaterEqual(len(times), 2)
        self.assertTrue(all(seg.start < t < seg.end for t in times))

    def test_change_selector_uses_local_peaks_and_spacing(self):
        cfg = IndexerConfig(
            semantic_diff_threshold=0.10,
            semantic_adaptive_margin=0.02,
            semantic_min_boundary_spacing_sec=4.0,
            max_semantic_clip_sec=12.0,
            semantic_extra_split_allowance=2,
        )
        shot = TimeRange(0.0, 30.0)
        probe_times = [float(i) for i in range(0, 31, 2)]
        # many above-threshold movements, but only a few real peaks should survive
        scores = [0.05, 0.12, 0.13, 0.30, 0.14, 0.12, 0.07, 0.28, 0.13, 0.12, 0.06, 0.31, 0.12, 0.11, 0.05]
        boundaries = select_change_boundaries(shot, probe_times, scores, cfg)
        self.assertLessEqual(len(boundaries), 4)  # ceil(30/12)-1 + allowance = 4
        self.assertTrue(all(b - a >= 4.0 for a, b in zip(boundaries, boundaries[1:])))

    def test_small_motion_noise_does_not_split(self):
        cfg = IndexerConfig(semantic_diff_threshold=0.13, semantic_adaptive_margin=0.04)
        shot = TimeRange(0.0, 20.0)
        probe_times = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        scores = [0.08, 0.09, 0.10, 0.11, 0.09, 0.10, 0.08, 0.11, 0.09, 0.08]
        self.assertEqual(select_change_boundaries(shot, probe_times, scores, cfg), [])
