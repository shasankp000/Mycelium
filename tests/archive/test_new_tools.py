"""
tests/tools/test_new_tools.py
==============================
Smoke tests for the four new predicate-type-aware MCP tools.

All external HTTP calls are mocked — tests run fully offline.

Coverage matrix
---------------
  temporal_search
    - happy path: Wikidata SPARQL returns bindings, DDG returns snippets
    - SPARQL failure: falls back to DDG-only, result still well-formed
    - total failure: both passes fail -> status=='empty'

  causal_search
    - happy path: SemanticScholar returns papers, OpenCitations returns chain
    - SemanticScholar failure: papers=[], provenance_chain=[], status=='empty'
    - OpenCitations failure: papers returned, provenance_chain=[], no raise

  statistical_search
    - happy path (WB match): World Bank returns data rows
    - happy path (OECD match): OECD returns series observations
    - no keyword match: WB + OECD skipped, DDG fallback fires
    - all passes fail: status=='empty'

  domain_store_search
    - stub contract: status=='stub', required keys present, warning logged

  _resolve_tool_call_result (run_workflow helper)
    - stub result triggers web_search re-issue, fallback_from annotated
    - non-domain_store_search tool returns None (evidence_finder handles it)
"""
from __future__ import annotations

import json
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import mycelium.pipeline.mcp_tools_server as _srv
from mycelium.pipeline.mcp_tools_server import (
    _temporal_search,
    _causal_search,
    _statistical_search,
    _domain_store_search,
    _TOOL_FN,
)

# Also import the run_workflow helper we want to test
from mycelium.pipeline.run_workflow import _resolve_tool_call_result


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

def _sparql_response(bindings: list) -> MagicMock:
    """Build a mock requests.Response for Wikidata SPARQL."""
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json.return_value = {"results": {"bindings": bindings}}
    return m


def _json_response(payload) -> MagicMock:
    """Generic mock requests.Response returning JSON payload."""
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json.return_value = payload
    return m


def _empty_ddgs(query, max_results=5):
    """Stub for _ddgs_search that returns nothing."""
    return []


def _nonempty_ddgs(query, max_results=5):
    """Stub for _ddgs_search that returns a single result."""
    return [{"title": "DDG result", "snippet": "some snippet", "url": "https://example.com"}]


# ---------------------------------------------------------------------------
# temporal_search
# ---------------------------------------------------------------------------

class TestTemporalSearch(unittest.TestCase):

    _SPARQL_BINDING = [
        {
            "itemLabel": {"value": "Berlin Wall"},
            "propLabel": {"value": "inception"},
            "date":      {"value": "1961-08-13T00:00:00Z"},
        }
    ]

    def test_happy_path_returns_wikidata_and_web_snippets(self):
        """
        When SPARQL succeeds and DDG returns snippets,
        result must contain non-empty wikidata_dates and web_snippets,
        status=='ok', and required top-level keys are present.
        """
        with patch.object(_srv._SESSION, "get", return_value=_sparql_response(self._SPARQL_BINDING)), \
             patch("mycelium.pipeline.mcp_tools_server._ddgs_search", side_effect=_nonempty_ddgs):
            result = _temporal_search("When did the Berlin Wall fall 1989")

        self.assertIn("status", result)
        self.assertEqual(result["status"], "ok")
        self.assertIn("wikidata_dates", result)
        self.assertTrue(len(result["wikidata_dates"]) > 0)
        self.assertIn("web_snippets", result)
        self.assertTrue(len(result["web_snippets"]) > 0)
        # Verify wikidata_dates entry shape
        entry = result["wikidata_dates"][0]
        for key in ("entity", "property", "date"):
            self.assertIn(key, entry)

    def test_sparql_failure_falls_back_to_ddg_only(self):
        """
        When SPARQL raises, result must still be well-formed;
        web_snippets must be non-empty (DDG fallback), no exception escapes.
        """
        with patch.object(_srv._SESSION, "get", side_effect=ConnectionError("SPARQL down")), \
             patch("mycelium.pipeline.mcp_tools_server._ddgs_search", side_effect=_nonempty_ddgs):
            result = _temporal_search("French Revolution 1789")

        self.assertIn("status", result)
        self.assertEqual(result["status"], "ok")  # DDG kept it alive
        self.assertEqual(result["wikidata_dates"], [])
        self.assertTrue(len(result["web_snippets"]) > 0)

    def test_total_failure_returns_empty_status(self):
        """
        When both SPARQL and DDG fail entirely,
        status must be 'empty' and no exception escapes.
        """
        with patch.object(_srv._SESSION, "get", side_effect=ConnectionError("offline")), \
             patch("mycelium.pipeline.mcp_tools_server._ddgs_search", side_effect=_empty_ddgs):
            result = _temporal_search("some obscure event 1750")

        self.assertIn("status", result)
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["wikidata_dates"], [])
        self.assertEqual(result["web_snippets"], [])
        self.assertIn("note", result)


# ---------------------------------------------------------------------------
# causal_search
# ---------------------------------------------------------------------------

_SS_PAPER = {
    "title": "Causal inference in observational studies",
    "year": 2019,
    "authors": [{"name": "A. Smith"}],
    "abstract": "We study causal effects using potential outcomes.",
    "citationCount": 450,
    "externalIds": {"DOI": "10.1000/test.doi"},
}

_OC_ENTRY = {
    "citing": "doi:10.1000/test.doi",
    "cited":  "doi:10.1000/other.doi",
    "creation": "2019",
}


class TestCausalSearch(unittest.TestCase):

    def test_happy_path_returns_papers_and_provenance(self):
        """
        When SemanticScholar and OpenCitations both succeed,
        papers must be non-empty with required keys, provenance_chain
        must be non-empty with required keys.
        """
        ss_resp = _json_response({"data": [_SS_PAPER]})
        oc_resp = _json_response([_OC_ENTRY])

        call_count = {"n": 0}

        def _mock_get(url, **kwargs):
            call_count["n"] += 1
            if "semanticscholar" in url:
                return ss_resp
            if "opencitations" in url:
                return oc_resp
            return _json_response({})

        with patch.object(_srv._SESSION, "get", side_effect=_mock_get):
            result = _causal_search("does smoking cause lung cancer")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(len(result["papers"]) > 0)
        paper = result["papers"][0]
        for key in ("title", "year", "authors", "abstract", "citations", "doi"):
            self.assertIn(key, paper)

        self.assertTrue(len(result["provenance_chain"]) > 0)
        chain_entry = result["provenance_chain"][0]
        for key in ("citing", "cited", "creation"):
            self.assertIn(key, chain_entry)

    def test_semantic_scholar_failure_returns_empty_status(self):
        """
        When SemanticScholar raises, papers=[], provenance_chain=[],
        status=='empty', no exception escapes.
        """
        with patch.object(_srv._SESSION, "get", side_effect=ConnectionError("SS down")):
            result = _causal_search("does exercise cause weight loss")

        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["papers"], [])
        self.assertEqual(result["provenance_chain"], [])
        self.assertIn("note", result)

    def test_opencitations_failure_does_not_affect_papers(self):
        """
        When SemanticScholar succeeds but OpenCitations raises,
        papers must still be populated, provenance_chain=[], no raise.
        """
        ss_resp = _json_response({"data": [_SS_PAPER]})

        def _mock_get(url, **kwargs):
            if "semanticscholar" in url:
                return ss_resp
            raise ConnectionError("OC down")

        with patch.object(_srv._SESSION, "get", side_effect=_mock_get):
            result = _causal_search("does diet affect blood pressure")

        self.assertIn("status", result)
        # papers still populated even if OC failed
        self.assertTrue(len(result["papers"]) > 0)
        self.assertEqual(result["provenance_chain"], [])
        # status should remain ok because papers were found
        self.assertEqual(result["status"], "ok")


# ---------------------------------------------------------------------------
# statistical_search
# ---------------------------------------------------------------------------

_WB_PAYLOAD = [
    {"page": 1, "pages": 1, "per_page": 5, "total": 5},
    [
        {"date": "2022", "value": 12345.6, "country": {"value": "World"},
         "indicator": {"id": "NY.GDP.PCAP.CD"}},
        {"date": "2021", "value": 11900.0, "country": {"value": "World"},
         "indicator": {"id": "NY.GDP.PCAP.CD"}},
    ],
]

_OECD_PAYLOAD = {
    "dataSets": [{
        "series": {
            "0:0:0": {
                "observations": {"0": [7.5]}
            }
        }
    }],
    "structure": {}
}


class TestStatisticalSearch(unittest.TestCase):

    def test_world_bank_happy_path(self):
        """
        Query containing 'gdp per capita' triggers WB lookup.
        world_bank must be non-empty with year/value/country/indicator keys.
        """
        wb_resp = _json_response(_WB_PAYLOAD)

        with patch.object(_srv._SESSION, "get", return_value=wb_resp), \
             patch("mycelium.pipeline.mcp_tools_server._ddgs_search", side_effect=_nonempty_ddgs):
            result = _statistical_search("gdp per capita data by country")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(len(result["world_bank"]) > 0)
        row = result["world_bank"][0]
        for key in ("year", "value", "country", "indicator"):
            self.assertIn(key, row)

    def test_oecd_happy_path(self):
        """
        Query containing 'health expenditure' triggers OECD lookup.
        oecd must be non-empty with series_key/period/value/dataflow keys.
        """
        oecd_resp = _json_response(_OECD_PAYLOAD)

        with patch.object(_srv._SESSION, "get", return_value=oecd_resp), \
             patch("mycelium.pipeline.mcp_tools_server._ddgs_search", side_effect=_nonempty_ddgs):
            result = _statistical_search("health expenditure across countries")

        self.assertEqual(result["status"], "ok")
        self.assertTrue(len(result["oecd"]) > 0)
        obs = result["oecd"][0]
        for key in ("series_key", "period", "value", "dataflow"):
            self.assertIn(key, obs)

    def test_no_keyword_match_falls_back_to_ddg(self):
        """
        Query with no WB/OECD keyword skips both API passes;
        web_snippets must be populated from DDG fallback.
        """
        with patch("mycelium.pipeline.mcp_tools_server._ddgs_search", side_effect=_nonempty_ddgs):
            result = _statistical_search("some random statistical claim")

        self.assertIn("status", result)
        self.assertEqual(result["world_bank"], [])
        self.assertEqual(result["oecd"], [])
        self.assertTrue(len(result["web_snippets"]) > 0)

    def test_all_passes_fail_returns_empty_status(self):
        """
        When WB, OECD, and DDG all fail,
        status must be 'empty' and no exception escapes.
        """
        with patch.object(_srv._SESSION, "get", side_effect=ConnectionError("offline")), \
             patch("mycelium.pipeline.mcp_tools_server._ddgs_search", side_effect=_empty_ddgs):
            result = _statistical_search("gdp per capita trend")

        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["world_bank"], [])
        self.assertEqual(result["oecd"], [])
        self.assertEqual(result["web_snippets"], [])
        self.assertIn("note", result)


# ---------------------------------------------------------------------------
# domain_store_search — stub contract
# ---------------------------------------------------------------------------

class TestDomainStoreSearchStub(unittest.TestCase):

    def test_stub_returns_required_keys(self):
        """
        domain_store_search must always return a dict with these keys:
        chunks, status, note, source, query.
        """
        result = _domain_store_search("what is photosynthesis")
        for key in ("chunks", "status", "note", "source", "query"):
            self.assertIn(key, result)

    def test_stub_status_is_stub(self):
        result = _domain_store_search("anything")
        self.assertEqual(result["status"], "stub")

    def test_stub_chunks_is_empty_list(self):
        result = _domain_store_search("anything")
        self.assertIsInstance(result["chunks"], list)
        self.assertEqual(len(result["chunks"]), 0)

    def test_stub_logs_warning(self):
        """domain_store_search must log a warning so traces surface the gap."""
        with self.assertLogs("mycelium.pipeline.mcp_tools_server", level="WARNING") as cm:
            _domain_store_search("test query")
        self.assertTrue(
            any("stub" in msg.lower() or "domain_store_search" in msg for msg in cm.output)
        )


# ---------------------------------------------------------------------------
# _resolve_tool_call_result — fallback policy (run_workflow helper)
# ---------------------------------------------------------------------------

def _make_tool_call(tool: str, query: str):
    """Minimal ToolCall-like object satisfying the duck-type in run_workflow."""
    obj = types.SimpleNamespace(tool=tool, query=query)
    return obj


class TestResolveToolCallResult(unittest.TestCase):

    def test_domain_store_stub_triggers_web_search_fallback(self):
        """
        When domain_store_search returns status=='stub',
        _resolve_tool_call_result must re-issue as web_search and
        annotate the result with fallback_from='domain_store_search'.
        """
        stub_result   = {"status": "stub", "note": "TRM v2 not built", "chunks": []}
        web_result    = {"status": "ok",   "results": [{"snippet": "found it"}]}

        mock_tool_fns = {
            "domain_store_search": MagicMock(return_value=stub_result),
            "web_search":          MagicMock(return_value=web_result),
        }

        with patch.dict("mycelium.pipeline.mcp_tools_server._TOOL_FN", mock_tool_fns), \
             patch("mycelium.pipeline.run_workflow._resolve_tool_call_result",
                   wraps=_resolve_tool_call_result):

            # Import the real function fresh to get the patched _TOOL_FN
            from mycelium.pipeline.run_workflow import _resolve_tool_call_result as _r
            tc = _make_tool_call("domain_store_search", "what is gravity")
            result = _r(tc, evidence_finder=MagicMock(), domain_hint="physics")

        self.assertIsNotNone(result)
        self.assertEqual(result.get("fallback_from"), "domain_store_search")
        self.assertEqual(result.get("status"), "ok")

    def test_non_domain_store_tool_returns_none(self):
        """
        For any tool other than domain_store_search,
        _resolve_tool_call_result must return None
        (evidence_finder handles those tools internally).
        """
        from mycelium.pipeline.run_workflow import _resolve_tool_call_result as _r
        for tool_name in ("web_search", "academic_search", "temporal_search", "causal_search"):
            tc = _make_tool_call(tool_name, "test query")
            result = _r(tc, evidence_finder=MagicMock(), domain_hint="general")
            self.assertIsNone(
                result,
                msg=f"Expected None for tool '{tool_name}', got {result}"
            )

    def test_none_tool_call_returns_none(self):
        from mycelium.pipeline.run_workflow import _resolve_tool_call_result as _r
        self.assertIsNone(_r(None, evidence_finder=MagicMock(), domain_hint="general"))

    def test_none_evidence_finder_returns_none(self):
        from mycelium.pipeline.run_workflow import _resolve_tool_call_result as _r
        tc = _make_tool_call("domain_store_search", "test")
        self.assertIsNone(_r(tc, evidence_finder=None, domain_hint="general"))


if __name__ == "__main__":
    unittest.main()
