import unittest

from speechlab_diarization.backends.labels import (
    CANONICAL_LABELS,
    LABEL_NONE,
    get_one_hot_probabilities,
    get_uniform_probabilities,
    normalize_label,
)


class TestNormalizeLabel(unittest.TestCase):
    # --- VTC1 ---
    def test_vtc1_fem(self):
        self.assertEqual(normalize_label("FEM", "vtc1"), "FEM")

    def test_vtc1_mal(self):
        self.assertEqual(normalize_label("MAL", "vtc1"), "MAL")

    def test_vtc1_kchi(self):
        self.assertEqual(normalize_label("KCHI", "vtc1"), "KCHI")

    def test_vtc1_chi_maps_to_och(self):
        self.assertEqual(normalize_label("CHI", "vtc1"), "OCH")

    def test_vtc1_oc_maps_to_och(self):
        self.assertEqual(normalize_label("OC", "vtc1"), "OCH")

    def test_vtc1_och_already_canonical(self):
        self.assertEqual(normalize_label("OCH", "vtc1"), "OCH")

    def test_vtc1_speech_maps_to_none(self):
        self.assertEqual(normalize_label("SPEECH", "vtc1"), LABEL_NONE)

    def test_vtc1_unknown_maps_to_none(self):
        self.assertEqual(normalize_label("XYZ", "vtc1"), LABEL_NONE)

    # --- VTC2 ---
    def test_vtc2_fem(self):
        self.assertEqual(normalize_label("FEM", "vtc2"), "FEM")

    def test_vtc2_mal(self):
        self.assertEqual(normalize_label("MAL", "vtc2"), "MAL")

    def test_vtc2_kchi(self):
        self.assertEqual(normalize_label("KCHI", "vtc2"), "KCHI")

    def test_vtc2_och(self):
        self.assertEqual(normalize_label("OCH", "vtc2"), "OCH")

    def test_vtc2_chi_not_mapped(self):
        self.assertEqual(normalize_label("CHI", "vtc2"), LABEL_NONE)

    def test_vtc2_unknown_maps_to_none(self):
        self.assertEqual(normalize_label("UNKNOWN", "vtc2"), LABEL_NONE)

    # --- case / whitespace ---
    def test_lowercase_input(self):
        self.assertEqual(normalize_label("fem", "vtc2"), "FEM")

    def test_mixed_case_input(self):
        self.assertEqual(normalize_label("Kchi", "vtc1"), "KCHI")

    def test_whitespace_stripped(self):
        self.assertEqual(normalize_label("  FEM  ", "vtc2"), "FEM")

    # --- unknown backend fallback ---
    def test_unknown_backend_canonical_passes_through(self):
        self.assertEqual(normalize_label("KCHI", "other"), "KCHI")

    def test_unknown_backend_noncanonical_maps_to_none(self):
        self.assertEqual(normalize_label("CHI", "other"), LABEL_NONE)


class TestGetOneHotProbabilities(unittest.TestCase):
    def test_target_label_gets_one(self):
        probs = get_one_hot_probabilities("FEM")
        self.assertEqual(probs["FEM"], 1.0)

    def test_other_labels_get_zero(self):
        probs = get_one_hot_probabilities("FEM")
        for label in CANONICAL_LABELS:
            if label != "FEM":
                self.assertEqual(probs[label], 0.0)

    def test_sums_to_one_for_valid_label(self):
        probs = get_one_hot_probabilities("KCHI")
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_unknown_label_all_zeros(self):
        probs = get_one_hot_probabilities("UNKNOWN")
        self.assertTrue(all(v == 0.0 for v in probs.values()))

    def test_all_canonical_labels_produce_valid_distribution(self):
        for label in CANONICAL_LABELS:
            probs = get_one_hot_probabilities(label)
            self.assertEqual(probs[label], 1.0)
            self.assertAlmostEqual(sum(probs.values()), 1.0)


class TestGetUniformProbabilities(unittest.TestCase):
    def test_all_labels_equal(self):
        probs = get_uniform_probabilities()
        values = list(probs.values())
        self.assertTrue(all(v == values[0] for v in values))

    def test_sums_to_one(self):
        probs = get_uniform_probabilities()
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_covers_all_canonical_labels(self):
        probs = get_uniform_probabilities()
        for label in CANONICAL_LABELS:
            self.assertIn(label, probs)

    def test_correct_value_per_label(self):
        probs = get_uniform_probabilities()
        expected = 1.0 / len(CANONICAL_LABELS)
        for v in probs.values():
            self.assertAlmostEqual(v, expected)


if __name__ == "__main__":
    unittest.main()
