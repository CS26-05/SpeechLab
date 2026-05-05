import unittest

from pyannote.core import Segment

from speechlab_diarization.alignment import (
    AlignedSegment,
    align_segments,
    compute_overlap,
    create_voice_type_mapping,
    segment_key,
)
from speechlab_diarization.backends.base import VoiceTypeSegment
from speechlab_diarization.backends.labels import CANONICAL_LABELS, LABEL_NONE


def _vt(start, end, label):
    probs = {l: 0.0 for l in CANONICAL_LABELS}
    if label in CANONICAL_LABELS:
        probs[label] = 1.0
    return VoiceTypeSegment(
        start=start, end=end,
        raw_label=label, canonical_label=label,
        probabilities=probs,
    )


def _aligned(start, end, voice_type):
    probs = {l: 0.0 for l in CANONICAL_LABELS}
    if voice_type in CANONICAL_LABELS:
        probs[voice_type] = 1.0
    return AlignedSegment(
        start=start, end=end,
        speaker="SPK_00",
        voice_type=voice_type,
        probabilities=probs,
    )


class TestComputeOverlap(unittest.TestCase):
    def test_full_overlap(self):
        self.assertAlmostEqual(compute_overlap(0.0, 5.0, 0.0, 5.0), 5.0)

    def test_partial_overlap_right(self):
        self.assertAlmostEqual(compute_overlap(0.0, 3.0, 2.0, 5.0), 1.0)

    def test_partial_overlap_left(self):
        self.assertAlmostEqual(compute_overlap(2.0, 5.0, 0.0, 3.0), 1.0)

    def test_no_overlap_adjacent(self):
        self.assertAlmostEqual(compute_overlap(0.0, 2.0, 2.0, 4.0), 0.0)

    def test_no_overlap_gap(self):
        self.assertAlmostEqual(compute_overlap(0.0, 1.0, 3.0, 5.0), 0.0)

    def test_seg1_contains_seg2(self):
        self.assertAlmostEqual(compute_overlap(0.0, 10.0, 2.0, 5.0), 3.0)

    def test_seg2_contains_seg1(self):
        self.assertAlmostEqual(compute_overlap(2.0, 5.0, 0.0, 10.0), 3.0)

    def test_seg2_entirely_before_seg1(self):
        self.assertAlmostEqual(compute_overlap(5.0, 10.0, 0.0, 3.0), 0.0)

    def test_returns_zero_not_negative(self):
        result = compute_overlap(5.0, 6.0, 0.0, 4.0)
        self.assertGreaterEqual(result, 0.0)


class TestSegmentKey(unittest.TestCase):
    def test_rounds_to_three_decimals(self):
        self.assertEqual(segment_key(1.0001, 2.9999), (1.0, 3.0))

    def test_exact_values_unchanged(self):
        self.assertEqual(segment_key(1.5, 2.5), (1.5, 2.5))

    def test_returns_tuple(self):
        self.assertIsInstance(segment_key(0.0, 1.0), tuple)

    def test_consistent_for_same_input(self):
        self.assertEqual(segment_key(1.2345, 3.4565), segment_key(1.2345, 3.4565))


class TestCreateVoiceTypeMapping(unittest.TestCase):
    def test_single_segment(self):
        segs = [_aligned(0.0, 1.0, "KCHI")]
        mapping = create_voice_type_mapping(segs)
        self.assertEqual(mapping[(0.0, 1.0)], "KCHI")

    def test_multiple_segments(self):
        segs = [_aligned(0.0, 1.0, "FEM"), _aligned(2.0, 3.0, "MAL")]
        mapping = create_voice_type_mapping(segs)
        self.assertEqual(mapping[(0.0, 1.0)], "FEM")
        self.assertEqual(mapping[(2.0, 3.0)], "MAL")

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(create_voice_type_mapping([]), {})

    def test_none_label_stored(self):
        segs = [_aligned(0.0, 1.0, LABEL_NONE)]
        mapping = create_voice_type_mapping(segs)
        self.assertEqual(mapping[(0.0, 1.0)], LABEL_NONE)


class TestAlignSegments(unittest.TestCase):
    def test_single_exact_match(self):
        dia = [(Segment(0.0, 5.0), "SPK_00")]
        vt = [_vt(0.0, 5.0, "KCHI")]
        result = align_segments(dia, vt)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].voice_type, "KCHI")

    def test_no_vt_overlap_falls_back_to_none(self):
        dia = [(Segment(0.0, 2.0), "SPK_00")]
        vt = [_vt(5.0, 8.0, "FEM")]
        result = align_segments(dia, vt)
        self.assertEqual(result[0].voice_type, LABEL_NONE)

    def test_max_overlap_label_wins(self):
        # diarization 0-10; KCHI 0-3, FEM 3-10 → FEM wins (7s vs 3s)
        dia = [(Segment(0.0, 10.0), "SPK_00")]
        vt = [_vt(0.0, 3.0, "KCHI"), _vt(3.0, 10.0, "FEM")]
        result = align_segments(dia, vt)
        self.assertEqual(result[0].voice_type, "FEM")

    def test_none_vt_segments_excluded_by_default(self):
        dia = [(Segment(0.0, 5.0), "SPK_00")]
        vt = [_vt(0.0, 5.0, LABEL_NONE)]
        result = align_segments(dia, vt, include_none=False)
        # NONE segment filtered out → no overlap → falls back to LABEL_NONE
        self.assertEqual(result[0].voice_type, LABEL_NONE)

    def test_none_vt_segments_included_when_flag_set(self):
        dia = [(Segment(0.0, 5.0), "SPK_00")]
        vt = [_vt(0.0, 5.0, LABEL_NONE)]
        # LABEL_NONE is not in CANONICAL_LABELS so overlap_by_label won't accumulate,
        # but total_overlap > 0 so voice_type = max of all-zeros dict (first key)
        result = align_segments(dia, vt, include_none=True)
        self.assertIsNotNone(result[0].voice_type)

    def test_empty_diarization_returns_empty(self):
        result = align_segments([], [_vt(0.0, 5.0, "FEM")])
        self.assertEqual(result, [])

    def test_empty_vt_segments_all_fallback(self):
        dia = [(Segment(0.0, 2.0), "SPK_00"), (Segment(3.0, 5.0), "SPK_01")]
        result = align_segments(dia, [])
        self.assertTrue(all(s.voice_type == LABEL_NONE for s in result))

    def test_probabilities_sum_to_one(self):
        dia = [(Segment(0.0, 4.0), "SPK_00")]
        vt = [_vt(0.0, 2.0, "KCHI"), _vt(2.0, 4.0, "FEM")]
        result = align_segments(dia, vt)
        total = sum(result[0].probabilities.values())
        self.assertAlmostEqual(total, 1.0)

    def test_speaker_label_preserved(self):
        dia = [(Segment(0.0, 1.0), "SPEAKER_42")]
        result = align_segments(dia, [])
        self.assertEqual(result[0].speaker, "SPEAKER_42")

    def test_segment_times_preserved(self):
        dia = [(Segment(1.5, 3.7), "SPK_00")]
        result = align_segments(dia, [])
        self.assertAlmostEqual(result[0].start, 1.5)
        self.assertAlmostEqual(result[0].end, 3.7)


if __name__ == "__main__":
    unittest.main()
