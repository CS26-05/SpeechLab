import tempfile
import unittest
from pathlib import Path

from pyannote.core import Annotation, Segment

from speechlab_diarization.rttm_io import (
    _safe_label,
    parse_enriched_rttm,
    segment_key,
    write_enriched_rttm,
    write_plain_rttm,
)


class TestSafeLabel(unittest.TestCase):
    def test_vtc_labels_pass_through(self):
        for label in ("FEM", "MAL", "KCHI", "OCH"):
            self.assertEqual(_safe_label(label), label)

    def test_none_label_passes_through(self):
        self.assertEqual(_safe_label("NONE"), "NONE")

    def test_lowercase_normalised(self):
        self.assertEqual(_safe_label("fem"), "FEM")
        self.assertEqual(_safe_label("kchi"), "KCHI")

    def test_unknown_label_returns_none(self):
        self.assertEqual(_safe_label("UNKNOWN"), "NONE")
        self.assertEqual(_safe_label("UNK"), "NONE")
        self.assertEqual(_safe_label("SPK_00"), "NONE")

    def test_empty_string_returns_none(self):
        self.assertEqual(_safe_label(""), "NONE")

    def test_none_value_returns_none(self):
        self.assertEqual(_safe_label(None), "NONE")

    def test_whitespace_stripped(self):
        self.assertEqual(_safe_label("  KCHI  "), "KCHI")


class TestSegmentKey(unittest.TestCase):
    def test_rounds_to_three_decimals(self):
        self.assertEqual(segment_key(1.0001, 2.9999), (1.0, 3.0))

    def test_exact_values_unchanged(self):
        self.assertEqual(segment_key(0.5, 1.5), (0.5, 1.5))

    def test_returns_tuple(self):
        self.assertIsInstance(segment_key(0.0, 1.0), tuple)
        self.assertEqual(len(segment_key(0.0, 1.0)), 2)


class TestParseEnrichedRttm(unittest.TestCase):
    def _write(self, content):
        tmp = tempfile.NamedTemporaryFile(suffix=".rttm", delete=False, mode="w", encoding="utf-8")
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_parses_single_segment(self):
        path = self._write("SPEAKER testfile 1 0.000 2.500 <NA> <NA> KCHI <NA> <NA>\n")
        segs = parse_enriched_rttm(path)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["uri"], "testfile")
        self.assertAlmostEqual(segs[0]["start"], 0.0)
        self.assertAlmostEqual(segs[0]["duration"], 2.5)
        self.assertEqual(segs[0]["voice_type"], "KCHI")

    def test_parses_multiple_segments(self):
        path = self._write(
            "SPEAKER f 1 0.000 1.000 <NA> <NA> KCHI <NA> <NA>\n"
            "SPEAKER f 1 1.000 2.000 <NA> <NA> FEM <NA> <NA>\n"
        )
        segs = parse_enriched_rttm(path)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["voice_type"], "KCHI")
        self.assertEqual(segs[1]["voice_type"], "FEM")

    def test_skips_blank_lines(self):
        path = self._write("\n\nSPEAKER f 1 0.000 1.000 <NA> <NA> FEM <NA> <NA>\n\n")
        segs = parse_enriched_rttm(path)
        self.assertEqual(len(segs), 1)

    def test_skips_comment_lines(self):
        path = self._write("# comment\nSPEAKER f 1 0.000 1.000 <NA> <NA> MAL <NA> <NA>\n")
        segs = parse_enriched_rttm(path)
        self.assertEqual(len(segs), 1)

    def test_skips_malformed_lines(self):
        path = self._write("SPEAKER too few\n")
        segs = parse_enriched_rttm(path)
        self.assertEqual(len(segs), 0)

    def test_empty_file_returns_empty_list(self):
        path = self._write("")
        segs = parse_enriched_rttm(path)
        self.assertEqual(segs, [])

    def test_result_dict_has_required_keys(self):
        path = self._write("SPEAKER f 1 0.000 1.000 <NA> <NA> OCH <NA> <NA>\n")
        segs = parse_enriched_rttm(path)
        self.assertIn("uri", segs[0])
        self.assertIn("start", segs[0])
        self.assertIn("duration", segs[0])
        self.assertIn("voice_type", segs[0])


class TestWriteEnrichedRttmRoundTrip(unittest.TestCase):
    def test_round_trip_labels_preserved(self):
        ann = Annotation(uri="test_file")
        ann[Segment(0.0, 2.0)] = "SPK_00"
        ann[Segment(3.0, 5.0)] = "SPK_01"
        mapping = {(0.0, 2.0): "KCHI", (3.0, 5.0): "FEM"}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_file.rttm"
            write_enriched_rttm(ann, "test_file", out_path, mapping)
            segs = parse_enriched_rttm(out_path)

        self.assertEqual(len(segs), 2)
        labels = {s["voice_type"] for s in segs}
        self.assertEqual(labels, {"KCHI", "FEM"})

    def test_round_trip_timings_preserved(self):
        ann = Annotation(uri="f")
        ann[Segment(1.5, 3.0)] = "SPK_00"
        mapping = {(1.5, 3.0): "MAL"}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "f.rttm"
            write_enriched_rttm(ann, "f", out_path, mapping)
            segs = parse_enriched_rttm(out_path)

        self.assertAlmostEqual(segs[0]["start"], 1.5)
        self.assertAlmostEqual(segs[0]["duration"], 1.5)

    def test_unmapped_segment_written_as_none(self):
        ann = Annotation(uri="f")
        ann[Segment(0.0, 1.0)] = "SPK_00"
        mapping = {}  # no entry for this segment

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "f.rttm"
            write_enriched_rttm(ann, "f", out_path, mapping)
            segs = parse_enriched_rttm(out_path)

        self.assertEqual(segs[0]["voice_type"], "NONE")

    def test_write_plain_rttm_creates_file(self):
        ann = Annotation(uri="plain_test")
        ann[Segment(0.0, 1.0)] = "SPEAKER_00"

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "plain_test.rttm"
            write_plain_rttm(ann, "plain_test", out_path)
            self.assertTrue(out_path.exists())
            content = out_path.read_text()
            self.assertIn("SPEAKER", content)


if __name__ == "__main__":
    unittest.main()
