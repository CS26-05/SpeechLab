import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import evaluate_der


class TestEvaluateDER(unittest.TestCase):
    def test_zero_der_when_ref_equals_hyp(self):
        # Make a temp folder with test_reference/ and test_output/
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            ref_dir = base / "test_reference"
            hyp_dir = base / "test_output"
            ref_dir.mkdir()
            hyp_dir.mkdir()

            # Point the script to these temp dirs
            evaluate_der.REFERENCE_DIR = ref_dir
            evaluate_der.HYPOTHESIS_DIR = hyp_dir

            # Same RTTM content for ref + hyp
            rttm = "SPEAKER file1 1 0.00 2.00 <NA> <NA> SPK1 <NA> <NA>\n"
            (ref_dir / "file1.rttm").write_text(rttm)
            (hyp_dir / "file1.rttm").write_text(rttm)

            # Run main() and capture its printed output
            buf = io.StringIO()
            with redirect_stdout(buf):
                evaluate_der.main()
            out = buf.getvalue()

            # Check that DER is zero for this file and overall
            self.assertIn("file1: DER = 0.0000", out)
            self.assertIn("Overall DER: 0.0000", out)


if __name__ == "__main__":
    unittest.main()
