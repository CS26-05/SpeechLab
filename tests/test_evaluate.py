import unittest

from pyannote.core import Annotation, Segment, Timeline

from speechlab_diarization.evaluate import (
    LABELS,
    _timeline_intersection,
    annotation_to_detection,
    compute_per_class_f1,
    filter_annotation,
)


def _ann(segments, uri="test"):
    ann = Annotation(uri=uri)
    for start, end, label in segments:
        ann[Segment(start, end)] = label
    return ann


def _tl(segments):
    return Timeline([Segment(s, e) for s, e in segments])


class TestFilterAnnotation(unittest.TestCase):
    def test_excluded_label_removed(self):
        ann = _ann([(0.0, 1.0, "UNK"), (1.0, 2.0, "FEM")])
        result = filter_annotation(ann, {"UNK"})
        self.assertNotIn("UNK", result.labels())
        self.assertIn("FEM", result.labels())

    def test_non_excluded_labels_all_kept(self):
        ann = _ann([(0.0, 1.0, "KCHI"), (1.0, 2.0, "FEM")])
        result = filter_annotation(ann, {"UNK"})
        self.assertEqual(set(result.labels()), {"KCHI", "FEM"})

    def test_multiple_excluded_labels(self):
        ann = _ann([(0.0, 1.0, "UNK"), (1.0, 2.0, "MED"), (2.0, 3.0, "KCHI")])
        result = filter_annotation(ann, {"UNK", "MED"})
        self.assertEqual(list(result.labels()), ["KCHI"])

    def test_all_excluded_returns_empty(self):
        ann = _ann([(0.0, 1.0, "UNK"), (1.0, 2.0, "MED")])
        result = filter_annotation(ann, {"UNK", "MED"})
        self.assertEqual(len(result), 0)

    def test_empty_annotation_returns_empty(self):
        ann = Annotation(uri="test")
        result = filter_annotation(ann, {"UNK"})
        self.assertEqual(len(result), 0)

    def test_empty_exclude_set_keeps_everything(self):
        ann = _ann([(0.0, 1.0, "KCHI"), (1.0, 2.0, "FEM")])
        result = filter_annotation(ann, set())
        self.assertEqual(set(result.labels()), {"KCHI", "FEM"})


class TestTimelineIntersection(unittest.TestCase):
    def test_non_overlapping_returns_zero(self):
        self.assertAlmostEqual(_timeline_intersection(_tl([(0.0, 1.0)]), _tl([(2.0, 3.0)])), 0.0)

    def test_fully_overlapping(self):
        self.assertAlmostEqual(_timeline_intersection(_tl([(0.0, 3.0)]), _tl([(0.0, 3.0)])), 3.0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(_timeline_intersection(_tl([(0.0, 2.0)]), _tl([(1.0, 3.0)])), 1.0)

    def test_empty_first_timeline(self):
        self.assertAlmostEqual(_timeline_intersection(_tl([]), _tl([(0.0, 1.0)])), 0.0)

    def test_empty_second_timeline(self):
        self.assertAlmostEqual(_timeline_intersection(_tl([(0.0, 1.0)]), _tl([])), 0.0)

    def test_multiple_segments_sum(self):
        tl1 = _tl([(0.0, 1.0), (2.0, 3.0)])
        tl2 = _tl([(0.0, 1.0), (2.0, 3.0)])
        self.assertAlmostEqual(_timeline_intersection(tl1, tl2), 2.0)


class TestComputePerClassF1(unittest.TestCase):
    def test_perfect_match_gives_f1_one(self):
        ref = _ann([(0.0, 2.0, "KCHI")])
        hyp = _ann([(0.0, 2.0, "KCHI")])
        scores = compute_per_class_f1(ref, hyp)
        self.assertAlmostEqual(scores["F1_KCHI"], 1.0)
        self.assertAlmostEqual(scores["P_KCHI"], 1.0)
        self.assertAlmostEqual(scores["R_KCHI"], 1.0)

    def test_no_hypothesis_gives_zero_f1_and_recall(self):
        ref = _ann([(0.0, 2.0, "FEM")])
        hyp = Annotation(uri="test")
        scores = compute_per_class_f1(ref, hyp)
        self.assertAlmostEqual(scores["F1_FEM"], 0.0)
        self.assertAlmostEqual(scores["R_FEM"], 0.0)

    def test_no_reference_gives_zero_f1_and_precision(self):
        ref = Annotation(uri="test")
        hyp = _ann([(0.0, 2.0, "FEM")])
        scores = compute_per_class_f1(ref, hyp)
        self.assertAlmostEqual(scores["F1_FEM"], 0.0)
        self.assertAlmostEqual(scores["P_FEM"], 0.0)

    def test_no_overlap_gives_zero_f1(self):
        ref = _ann([(0.0, 1.0, "KCHI")])
        hyp = _ann([(5.0, 6.0, "KCHI")])
        scores = compute_per_class_f1(ref, hyp)
        self.assertAlmostEqual(scores["F1_KCHI"], 0.0)

    def test_avg_f1_is_mean_of_four_classes(self):
        ref = _ann([(0.0, 1.0, "KCHI")])
        hyp = _ann([(0.0, 1.0, "KCHI")])
        scores = compute_per_class_f1(ref, hyp)
        expected = sum(scores[f"F1_{l}"] for l in LABELS) / len(LABELS)
        self.assertAlmostEqual(scores["AvgF1"], expected, places=4)

    def test_all_metric_keys_present(self):
        ref = Annotation(uri="test")
        hyp = Annotation(uri="test")
        scores = compute_per_class_f1(ref, hyp)
        for label in LABELS:
            self.assertIn(f"F1_{label}", scores)
            self.assertIn(f"P_{label}", scores)
            self.assertIn(f"R_{label}", scores)
        self.assertIn("AvgF1", scores)

    def test_partial_overlap_precision_and_recall(self):
        # ref: KCHI 0-4, hyp: KCHI 2-6 → intersection=2, ref_dur=4, hyp_dur=4
        ref = _ann([(0.0, 4.0, "KCHI")])
        hyp = _ann([(2.0, 6.0, "KCHI")])
        scores = compute_per_class_f1(ref, hyp)
        self.assertAlmostEqual(scores["P_KCHI"], 0.5, places=3)
        self.assertAlmostEqual(scores["R_KCHI"], 0.5, places=3)

    def test_empty_both_gives_zero_avg_f1(self):
        ref = Annotation(uri="test")
        hyp = Annotation(uri="test")
        scores = compute_per_class_f1(ref, hyp)
        self.assertAlmostEqual(scores["AvgF1"], 0.0)


class TestAnnotationToDetection(unittest.TestCase):
    def test_all_segments_become_speech(self):
        ann = _ann([(0.0, 1.0, "KCHI"), (2.0, 3.0, "FEM")])
        det = annotation_to_detection(ann)
        for _, _, label in det.itertracks(yield_label=True):
            self.assertEqual(label, "speech")

    def test_segment_count_preserved(self):
        ann = _ann([(0.0, 1.0, "KCHI"), (2.0, 3.0, "FEM"), (4.0, 5.0, "MAL")])
        det = annotation_to_detection(ann)
        self.assertEqual(len(det), 3)

    def test_segment_times_preserved(self):
        ann = _ann([(1.5, 3.0, "MAL")])
        det = annotation_to_detection(ann)
        segs = [s for s, _, _ in det.itertracks(yield_label=True)]
        self.assertAlmostEqual(segs[0].start, 1.5)
        self.assertAlmostEqual(segs[0].end, 3.0)

    def test_empty_annotation_returns_empty(self):
        ann = Annotation(uri="test")
        det = annotation_to_detection(ann)
        self.assertEqual(len(det), 0)


if __name__ == "__main__":
    unittest.main()
