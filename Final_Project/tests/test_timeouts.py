import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app


TEXT = "Industrial machines transformed manufacturing and increased production across growing cities."
TEXT_TWO = TEXT + " Tropical butterflies display colorful wings throughout the summer breeding season."


def response(results=None, status=200):
    result = Mock(status_code=status)
    result.json.return_value = {"results": [] if results is None else results}
    return result


class TimeoutTests(unittest.TestCase):
    def test_timeout_retries_with_lighter_request(self):
        progress = Mock()
        with patch.object(app.requests, "post", side_effect=[app.requests.Timeout(), response()]) as post, patch.object(app.time, "sleep"):
            result = app.search_published_sources(TEXT, "dummy", progress)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["json"]["include_raw_content"], "text")
        self.assertFalse(post.call_args_list[1].kwargs["json"]["include_raw_content"])
        self.assertEqual(result["snippet_queries"], 1)
        self.assertFalse(result["incomplete"])
        self.assertEqual(len(result["excerpts"]), 1)  # Empty successful search is still completed.
        self.assertTrue(result["warnings"])
        self.assertTrue(progress.called)

    def test_no_retry_for_auth_quota_or_invalid_input(self):
        for status in (400, 401, 403, 422, 429, 432, 433):
            result = response(status=status)
            result.raise_for_status.side_effect = app.requests.HTTPError()
            with self.subTest(status=status), patch.object(app.requests, "post", return_value=result) as post:
                with self.assertRaises(app.TavilySearchError):
                    app.search_published_sources(TEXT_TWO, "dummy")
                self.assertEqual(post.call_count, 1)

    def test_partial_results_keep_evidence(self):
        source = {"title": "Source", "url": "https://example.org", "raw_content": TEXT}
        with patch.object(app.requests, "post", side_effect=[response([source]), app.requests.Timeout(), app.requests.Timeout()]), patch.object(app.time, "sleep"):
            result = app.search_published_sources(TEXT_TWO, "dummy")
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["requested_count"], 2)
        self.assertEqual(len(result["excerpts"]), 1)
        self.assertEqual(result["sources"][0]["content"], TEXT)

    def test_all_failures_produce_no_assessment(self):
        with patch.object(app.requests, "post", side_effect=app.requests.Timeout()) as post, patch.object(app.time, "sleep"):
            with self.assertRaisesRegex(app.TavilySearchError, "No searches completed"):
                app.search_published_sources(TEXT_TWO + " Modern computer systems store digital files in distributed cloud infrastructure.", "dummy")
        self.assertEqual(post.call_count, 4)  # Two attempts each, stop after two failed phrases.

    def test_budget_expiry_prevents_retry(self):
        expired = app.SEARCH_TIME_BUDGET_SECONDS + 1
        with patch.object(app.time, "monotonic", side_effect=[0, 0, expired, expired]), patch.object(app.time, "sleep"), patch.object(app.requests, "post", side_effect=app.requests.Timeout()) as post:
            with self.assertRaisesRegex(app.TavilySearchError, "time budget"):
                app.search_published_sources(TEXT, "dummy")
        self.assertEqual(post.call_count, 1)

    def test_longer_timeout_allows_slower_source_extraction(self):
        with patch.object(app.requests, "post", return_value=response()) as post:
            app.search_published_sources(TEXT, "dummy")
        self.assertEqual(post.call_args.kwargs["timeout"], (10, 90))
        self.assertEqual(app.SEARCH_TIME_BUDGET_SECONDS, 300)

    def test_transient_server_error_recovers(self):
        with patch.object(app.requests, "post", side_effect=[response(status=503), response()]) as post, patch.object(app.time, "sleep"):
            result = app.search_published_sources(TEXT, "dummy")
        self.assertEqual(post.call_count, 2)
        self.assertFalse(result["incomplete"])

    def test_full_text_is_not_replaced_by_snippet(self):
        full = {"url": "https://example.org", "raw_content": TEXT}
        snippet = {"url": "https://example.org", "content": "Short snippet"}
        for pair in ([full, snippet], [snippet, full]):
            with patch.object(app.requests, "post", side_effect=[response([pair[0]]), response([pair[1]])]):
                result = app.search_published_sources(TEXT_TWO, "dummy")
            self.assertEqual(result["sources"][0]["content"], TEXT)
            self.assertEqual(result["sources"][0]["content_kind"], "Page text")

    def test_old_history_migrates_and_partial_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "history.sqlite3"
            with closing(sqlite3.connect(db)) as connection:
                connection.execute("CREATE TABLE scans (id INTEGER PRIMARY KEY, created TEXT, name TEXT, words INTEGER, sources INTEGER, queries INTEGER, coverage REAL, flagged INTEGER, threshold REAL)")
                connection.execute("INSERT INTO scans (name) VALUES ('Existing scan')")
                connection.commit()
            with patch.object(app, "HISTORY_DB", db):
                app.save_scan(TEXT, {"sources": [], "excerpts": [{}], "requested_count": 2, "incomplete": True, "snippet_queries": 1}, "Partial scan", 0.5)
                with app.history_connection() as connection:
                    rows = connection.execute("SELECT name,status,queries,requested,coverage FROM scans ORDER BY id").fetchall()
            self.assertEqual(rows[0][:2], ("Existing scan", "legacy"))
            self.assertEqual(rows[1], ("Partial scan", "partial", 1, 2, None))


if __name__ == "__main__":
    unittest.main()
