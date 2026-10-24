import hashlib
import unittest
from unittest.mock import patch

import app


def source(text, url="https://example.org", kind="Page text"):
    return {"title": "Example source", "url": url, "content": text,
            "content_kind": kind, "description": text, "matched_excerpts": []}


class WordOverlapTests(unittest.TestCase):
    def test_ten_percent_counts_only_matching_words(self):
        copied = "solar panels convert sunlight into electricity using semiconductor materials efficiently"
        original = " ".join(f"original{i}" for i in range(90))
        text = copied + " " + original
        result = app.detected_word_overlap(text, [source(copied)])
        self.assertEqual(result["matched_words"], 10)
        self.assertEqual(result["total_words"], 100)
        self.assertAlmostEqual(result["percentage"], 10)
        self.assertEqual(text[result["spans"][0][0]:result["spans"][0][1]], copied)

    def test_overlapping_sources_do_not_double_count(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        result = app.detected_word_overlap(text, [source("alpha beta gamma delta epsilon zeta eta"),
                                                source("delta epsilon zeta eta theta iota kappa", "https://second.org")])
        self.assertEqual(result["matched_words"], 10)
        self.assertEqual(result["percentage"], 100)
        self.assertEqual(result["spans"], [[0, len(text)]])
        self.assertEqual(len({m["source_url"] for m in result["matches"]}), 2)

    def test_case_punctuation_and_original_layout_preserved(self):
        text = "HELLO, world!\n\nThese machines work efficiently."
        result = app.detected_word_overlap(text, [source("hello world these machines work efficiently")])
        self.assertEqual(result["percentage"], 100)
        rendered = app.highlight_word_spans(text, result["spans"])
        self.assertIn("HELLO, world!\n\nThese machines work efficiently", rendered)

    def test_short_coincidence_does_not_count(self):
        result = app.detected_word_overlap("one two three four unique", [source("one two three four different")])
        self.assertEqual(result["percentage"], 0)
        self.assertEqual(result["spans"], [])

    def test_no_evidence_is_unknown_not_zero(self):
        for sources in ([], [source("just three words")]):
            self.assertIsNone(app.detected_word_overlap("Some document words to check today.", sources)["percentage"])

    def test_cited_words_remain_matches_for_review(self):
        text = '"Solar panels convert sunlight into electricity" (Smith, 2025).'
        result = app.detected_word_overlap(text, [source("Solar panels convert sunlight into electricity")])
        self.assertEqual(result["matched_words"], 6)
        self.assertEqual(result["total_words"], 8)

    def test_page_text_preferred_over_snippet_and_highlights_source(self):
        text = "Solar panels convert sunlight into electricity"
        result = app.detected_word_overlap(text, [source(text, kind="Search excerpt only"),
                                                source("Before " + text + " after", "https://page.org")])
        match = result["matches"][0]
        self.assertEqual(match["source_url"], "https://page.org")
        start, end = match["source_highlight"]
        self.assertEqual(match["source_context"][start:end], text)

    def test_highlights_escape_document_markup(self):
        text = '<script>alert("secret")</script> alpha beta gamma delta epsilon'
        result = app.detected_word_overlap(text, [source(text)])
        rendered = app.highlight_word_spans(text, result["spans"])
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;", rendered)
        self.assertIn("<mark", rendered)

    def test_score_and_highlights_render_in_both_themes(self):
        from streamlit.testing.v1 import AppTest
        text = "solar panels convert sunlight into electricity using semiconductor materials efficiently"
        text += " " + " ".join(f"original{i}" for i in range(90))
        sources = [source("solar panels convert sunlight into electricity using semiconductor materials efficiently")]
        at = AppTest.from_file(app.__file__).run(timeout=60)
        at.text_area(key="susp_text").set_value(text)
        at.session_state["search_results"] = {"sources": sources, "excerpts": [],
                                              "document_hash": hashlib.sha256(text.encode()).hexdigest()}
        for dark in (False, True):
            at.toggle(key="dark_mode").set_value(dark).run(timeout=60)
            self.assertFalse(at.exception)
            self.assertTrue(any(m.label == "Estimated plagiarism" and m.value == "10.0%" for m in at.metric))
            self.assertTrue(any("<mark" in m.value for m in at.markdown))


if __name__ == "__main__":
    unittest.main()
