import unittest
from unittest.mock import Mock, patch
import app

class EngineTests(unittest.TestCase):
    def source(self, text):
        return dict(title="Example", url="https://example.org", description=text,
                    content=text, content_kind="Page text", matched_excerpts=[])

    def test_exact_and_unrelated(self):
        original = "Industrial machines transformed manufacturing and increased production across growing cities."
        rows = app.compare_web_sources(original + " Tropical butterflies display colorful wings throughout the summer breeding season.", [self.source(original)])
        self.assertAlmostEqual(rows[0]["TF-IDF cosine"], 1.0)
        self.assertLess(rows[1]["TF-IDF cosine"], 0.2)

    def test_sidebar_key_input(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(app.__file__).run(timeout=60)
        self.assertFalse(at.exception)
        self.assertEqual(at.text_input(key="tavily_key").label, "Tavily API key")
        self.assertEqual(at.slider(key="review_threshold").value, 0.5)

    def test_persistent_history_and_charts(self):
        import tempfile
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(app, "HISTORY_DB", Path(folder) / "history.sqlite3"):
                text = "Industrial machines transformed manufacturing and increased production across growing cities."
                app.save_scan(text, {"sources": [self.source(text)], "excerpts": []}, "Draft one", 0.5)
                app.save_scan(text, {"sources": [], "excerpts": []}, "No evidence", 0.5)
                with app.history_connection() as connection:
                    rows = connection.execute("SELECT name,coverage FROM scans ORDER BY id").fetchall()
                self.assertEqual(rows[0], ("Draft one", 100.0))
                self.assertIsNone(rows[1][1])
                at = AppTest.from_string("import app; app.show_history()").run(timeout=60)
                self.assertFalse(at.exception)
                self.assertEqual(len(at.get("plotly_chart")), 2)

    def test_empty_sources(self):
        self.assertEqual(app.compare_web_sources("Some text.", []), [])

    def test_theme_switch_preserves_draft_key_and_results(self):
        import hashlib
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(app.__file__).run(timeout=60)
        text = "Industrial machines transformed manufacturing and increased production across growing cities."
        results = {"sources": [self.source(text)], "excerpts": [],
                   "document_hash": hashlib.sha256(text.encode()).hexdigest()}
        at.session_state["search_results"] = results
        at.text_area(key="susp_text").set_value(text)
        at.text_input(key="tavily_key").set_value("dummy-local-test-key")
        for dark in (False, True):
            at.toggle(key="dark_mode").set_value(dark).run(timeout=60)
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["susp_text"], text)
            self.assertEqual(at.session_state["tavily_key"], "dummy-local-test-key")
            self.assertEqual(at.session_state["search_results"], results)
            self.assertTrue(at.get("plotly_chart"))
            expected = "#10100e" if dark else "#f7f8f2"
            self.assertTrue(any(f".stApp {{background:{expected}" in m.value for m in at.markdown))

    def test_provider_payload_and_deduplication(self):
        response = Mock(status_code=200)
        response.json.return_value = {"results": [{"url": "https://example.org", "content": "Eight words provide sufficient text for this test sentence.", "raw_content": None}]}
        with patch.object(app.requests, "post", return_value=response) as post:
            result = app.search_published_sources("Industrial machines transformed manufacturing and increased production across growing cities. Tropical butterflies display colorful wings throughout the summer breeding season.", "test-key")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["content_kind"], "Search excerpt only")
        self.assertEqual(len(result["sources"][0]["matched_excerpts"]), 2)
        self.assertEqual(post.call_args.kwargs["json"]["search_depth"], "basic")
        self.assertFalse(post.call_args.kwargs["json"]["include_answer"])

    def test_auth_quota_and_malformed(self):
        for status in (401, 429, 432, 433, 500):
            response = Mock(status_code=status)
            response.raise_for_status.side_effect = app.requests.HTTPError()
            with patch.object(app.requests, "post", return_value=response):
                with self.assertRaises(app.TavilySearchError):
                    app._request_tavily_search("test", "secret")
        with self.assertRaises(app.TavilySearchError):
            app._parse_tavily_results(Mock(json=lambda: {"results": None}))

if __name__ == "__main__":
    unittest.main()
