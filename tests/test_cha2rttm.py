import sys
import unittest
from pathlib import Path

# scripts/ is not an installable package — inject it into the path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from cha2rttm import (
    _extract_last_time_mark_ms,
    _parse_comment_offset_ms,
    iter_chat_utterances,
    rttm_line,
)


def _mark(start, end):
    return f"\x15{start}_{end}\x15"


class TestRttmLine(unittest.TestCase):
    def test_correct_format(self):
        line = rttm_line("file1", 1.5, 2.0, "KCHI")
        self.assertEqual(line, "SPEAKER file1 1 1.500 2.000 <NA> <NA> KCHI <NA> <NA>")

    def test_zero_start(self):
        line = rttm_line("test", 0.0, 1.234, "FEM")
        self.assertTrue(line.startswith("SPEAKER test 1 0.000 1.234"))

    def test_three_decimal_places(self):
        line = rttm_line("f", 0.001, 0.999, "MAL")
        self.assertIn("0.001", line)
        self.assertIn("0.999", line)

    def test_custom_channel(self):
        line = rttm_line("f", 0.0, 1.0, "OCH", chan=2)
        parts = line.split()
        self.assertEqual(parts[2], "2")

    def test_ten_fields(self):
        line = rttm_line("f", 0.0, 1.0, "KCHI")
        self.assertEqual(len(line.split()), 10)


class TestParseCommentOffsetMs(unittest.TestCase):
    def test_finds_offset(self):
        text = "@Comment: recording start at 5000\n"
        self.assertEqual(_parse_comment_offset_ms(text), 5000)

    def test_returns_none_when_absent(self):
        self.assertIsNone(_parse_comment_offset_ms("no offset here"))

    def test_case_insensitive(self):
        text = "@Comment: Recording Start At 1200\n"
        self.assertEqual(_parse_comment_offset_ms(text), 1200)

    def test_empty_string_returns_none(self):
        self.assertIsNone(_parse_comment_offset_ms(""))


class TestExtractLastTimeMarkMs(unittest.TestCase):
    def test_single_mark(self):
        result = _extract_last_time_mark_ms(f"hello {_mark(1000, 2000)}")
        self.assertEqual(result, (1000, 2000))

    def test_returns_last_of_multiple_marks(self):
        text = f"{_mark(0, 1000)} word {_mark(1000, 2000)}"
        result = _extract_last_time_mark_ms(text)
        self.assertEqual(result, (1000, 2000))

    def test_no_marks_returns_none(self):
        self.assertIsNone(_extract_last_time_mark_ms("no marks here"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_extract_last_time_mark_ms(""))

    def test_returns_tuple_of_ints(self):
        result = _extract_last_time_mark_ms(_mark(500, 1500))
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)


class TestIterChatUtterances(unittest.TestCase):
    def test_single_utterance(self):
        text = f"*CHI: hello {_mark(1000, 2000)}"
        result = list(iter_chat_utterances(text))
        self.assertEqual(len(result), 1)
        spk, start, end = result[0]
        self.assertEqual(spk, "CHI")
        self.assertEqual(start, 1000)
        self.assertEqual(end, 2000)

    def test_multiple_speakers(self):
        text = f"*CHI: hi {_mark(0, 1000)}\n*MOT: hello {_mark(1000, 2000)}\n"
        result = list(iter_chat_utterances(text))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "CHI")
        self.assertEqual(result[1][0], "MOT")

    def test_utterance_without_time_mark_skipped(self):
        text = f"*CHI: no timestamp\n*MOT: hi {_mark(1000, 2000)}\n"
        result = list(iter_chat_utterances(text))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "MOT")

    def test_empty_text_returns_empty(self):
        result = list(iter_chat_utterances(""))
        self.assertEqual(result, [])

    def test_continuation_line_included_in_block(self):
        # % tier line is a continuation — time mark on it should be found
        text = f"*CHI: word\n%com: {_mark(500, 1500)}\n"
        result = list(iter_chat_utterances(text))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 500)

    def test_start_and_end_times_correct(self):
        text = f"*FAT: bye {_mark(3000, 4500)}\n"
        result = list(iter_chat_utterances(text))
        _, start, end = result[0]
        self.assertEqual(start, 3000)
        self.assertEqual(end, 4500)

    def test_speaker_code_parsed_correctly(self):
        text = f"*MOT: hello {_mark(0, 1000)}\n"
        result = list(iter_chat_utterances(text))
        self.assertEqual(result[0][0], "MOT")


if __name__ == "__main__":
    unittest.main()
