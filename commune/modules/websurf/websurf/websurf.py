# websurf.py
# A self-contained web surfing module that:
# 1) queries multiple search engines (APIs if available, HTML fallback),
# 2) visits top results,
# 3) extracts/aggregates relevant text + links for LLM context building.
#
# Notes:
# - API engines are used only if their environment variables are present.
# - Fallback engine (DuckDuckGo HTML) works without API keys.
# - Be polite: built-in rate limiting, timeouts, and basic deduplication.
# - No external framework dependencies beyond: requests, bs4, lxml (optional), and standard lib.

import os
import re
import time
import json
import html
import math
import logging
import mimetypes
import threading
from urllib.parse import urljoin, urlparse, quote_plus

import requests
from bs4 import BeautifulSoup

# Optional: improve parsing if lxml is installed; otherwise html.parser will be used
DEFAULT_PARSER = "lxml"

# ------------ Utilities ------------

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def is_probably_binary(content_type: str) -> bool:
    if not content_type:
        return False
    ct = content_type.split(";")[0].strip().lower()
    if ct.startswith("text/"):
        return False
    # Consider common non-text types as binary
    return ct not in {
        "text/html",
        "application/xml",
        "application/xhtml+xml",
        "application/json",
        "text/plain",
        "application/rss+xml",
        "application/atom+xml",
    }

def clean_url(u: str) -> str:
    try:
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            return ""
        return f"{p.scheme}://{p.netloc}{p.path}" + (f"?{p.query}" if p.query else "")
    except Exception:
        return ""

def domain(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())

def score_relevance(text: str, query_terms):
    # Simple TF-based scoring with log dampening
    if not text:
        return 0.0
    toks = tokenize(text)
    if not toks:
        return 0.0
    tf = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for q in query_terms:
        if q in tf:
            score += 1.0 + math.log(1 + tf[q])
    # Bonus for phrase occurrences (rough)
    phrase = " ".join(query_terms)
    if phrase and phrase in text.lower():
        score *= 1.2
    return score

def truncate(s: str, n: int) -> str:
    s = s or ""
    return (s[: n - 1] + "…") if len(s) > n else s

# ------------ Search Providers ------------

class SearchAggregator:
    """
    Aggregates results from multiple engines.
    Engines used (if configured via env):
      - Google Custom Search JSON API: GOOGLE_API_KEY + GOOGLE_CSE_ID
      - Bing Web Search API: BING_API_KEY
      - Brave Search API: BRAVE_API_KEY
      - SerpAPI (Google): SERPAPI_KEY
      - SearXNG: SEARXNG_ENDPOINT
    Fallback:
      - DuckDuckGo HTML scraping (no key required)
    """

    def __init__(self, session: requests.Session | None = None, user_agent: str | None = None):
        self.session = session or requests.Session()
        self.headers = {
            "User-Agent": user_agent
            or os.getenv(
                "WEBSURF_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Safari/537.36",
            )
        }

    # --- Individual engines ---

    def google_cse(self, query: str, n: int = 10):
        key = os.getenv("GOOGLE_API_KEY")
        cx = os.getenv("GOOGLE_CSE_ID")
        if not key or not cx:
            return []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": key, "cx": cx, "q": query, "num": min(n, 10)}
        try:
            r = self.session.get(url, params=params, headers=self.headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            out = []
            for item in data.get("items", []):
                link = clean_url(item.get("link", ""))
                if link:
                    out.append(
                        {
                            "title": norm_space(item.get("title", "")),
                            "url": link,
                            "snippet": norm_space(item.get("snippet", "")),
                            "engine": "google_cse",
                        }
                    )
            return out
        except Exception:
            return []

    def bing(self, query: str, n: int = 10):
        key = os.getenv("BING_API_KEY")
        if not key:
            return []
        url = "https://api.bing.microsoft.com/v7.0/search"
        params = {"q": query, "count": min(n, 50), "responseFilter": "Webpages"}
        headers = dict(self.headers)
        headers["Ocp-Apim-Subscription-Key"] = key
        try:
            r = self.session.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            web = data.get("webPages", {}).get("value", [])
            out = []
            for item in web:
                link = clean_url(item.get("url", ""))
                if link:
                    out.append(
                        {
                            "title": norm_space(item.get("name", "")),
                            "url": link,
                            "snippet": norm_space(item.get("snippet", "")),
                            "engine": "bing",
                        }
                    )
            return out
        except Exception:
            return []

    def brave(self, query: str, n: int = 10):
        key = os.getenv("BRAVE_API_KEY")
        if not key:
            return []
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": min(n, 20), "source": "web"}
        headers = dict(self.headers)
        headers["Accept"] = "application/json"
        headers["X-Subscription-Token"] = key
        try:
            r = self.session.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            out = []
            for item in data.get("web", {}).get("results", []):
                link = clean_url(item.get("url", ""))
                if link:
                    out.append(
                        {
                            "title": norm_space(item.get("title", "")),
                            "url": link,
                            "snippet": norm_space(item.get("description", "")),
                            "engine": "brave",
                        }
                    )
            return out
        except Exception:
            return []

    def serpapi(self, query: str, n: int = 10):
        key = os.getenv("SERPAPI_KEY")
        if not key:
            return []
        url = "https://serpapi.com/search.json"
        params = {"q": query, "engine": "google", "api_key": key, "num": min(n, 10)}
        try:
            r = self.session.get(url, params=params, headers=self.headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            out = []
            for item in data.get("organic_results", []):
                link = clean_url(item.get("link", ""))
                if link:
                    out.append(
                        {
                            "title": norm_space(item.get("title", "")),
                            "url": link,
                            "snippet": norm_space(item.get("snippet", "")),
                            "engine": "serpapi",
                        }
                    )
            return out
        except Exception:
            return []

    def searxng(self, query: str, n: int = 10):
        endpoint = os.getenv("SEARXNG_ENDPOINT")
        if not endpoint:
            return []
        url = endpoint.rstrip("/") + "/search"
        params = {"q": query, "format": "json", "categories": "general", "language": "en", "safesearch": 0}
        try:
            r = self.session.get(url, params=params, headers=self.headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            out = []
            for item in data.get("results", [])[:n]:
                link = clean_url(item.get("url", ""))
                if link:
                    out.append(
                        {
                            "title": norm_space(item.get("title", "")),
                            "url": link,
                            "snippet": norm_space(item.get("content", "")),
                            "engine": "searxng",
                        }
                    )
            return out
        except Exception:
            return []

    def ddg_html(self, query: str, n: int = 10):
        # DuckDuckGo HTML fallback
        url = "https://html.duckduckgo.com/html"
        try:
            r = self.session.post(
                url,
                data={"q": query},
                headers=self.headers,
                timeout=15,
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, DEFAULT_PARSER if DEFAULT_PARSER in ("lxml", "html.parser") else "html.parser")
            out = []
            for a in soup.select("a.result__a")[:n]:
                link = clean_url(a.get("href"))
                title = norm_space(a.get_text(" "))
                snippet_el = a.find_parent("div", class_="result__body")
                snippet = ""
                if snippet_el:
                    sn = snippet_el.select_one(".result__snippet")
                    if sn:
                        snippet = norm_space(sn.get_text(" "))
                if link:
                    out.append({"title": title, "url": link, "snippet": snippet, "engine": "duckduckgo"})
            return out
        except Exception:
            return []

    def search(self, query: str, n_per_engine: int = 8, use_engines: list[str] | None = None):
        engines = use_engines or [
            "google_cse",
            "bing",
            "brave",
            "serpapi",
            "searxng",
            "ddg_html",
        ]
        results = []
        for eng in engines:
            try:
                fn = getattr(self, eng)
                batch = fn(query, n=n_per_engine)
                results.extend(batch)
                time.sleep(0.3)  # polite stagger
            except Exception:
                continue

        # Dedupe by URL
        seen = set()
        deduped = []
        for r in results:
            u = r.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            deduped.append(r)

        return deduped

# ------------ Page Fetch & Parse ------------

class PageFetcher:
    def __init__(self, user_agent: str | None = None, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent
                or os.getenv(
                    "WEBSURF_USER_AGENT",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Safari/537.36",
                )
            }
        )
        self.timeout = timeout

    def get(self, url: str):
        r = self.session.get(url, timeout=self.timeout, allow_redirects=True)
        r.raise_for_status()
        return r

class ContentExtractor:
    """
    Lightweight content extractor (readability-ish heuristics):
    - Prioritize <article>, then main content containers.
    - Fall back to biggest block of <p>/<div> text.
    """

    MAIN_SELECTORS = [
        "article",
        "main",
        "[role='main']",
        "div#main",
        "div.article",
        "section.article",
        "div.post",
        "div.entry-content",
    ]

    DROP_SELECTORS = [
        "nav",
        "aside",
        "header",
        "footer",
        ".sidebar",
        ".advert",
        ".ad",
        ".promo",
        ".cookie",
        ".newsletter",
        ".subscribe",
        ".modal",
        ".popup",
        ".social",
        ".share",
        ".breadcrumb",
        ".comment",
        ".related",
        ".recommend",
    ]

    def __init__(self):
        self.parser = DEFAULT_PARSER if DEFAULT_PARSER in ("lxml", "html.parser") else "html.parser"

    def _strip_noise(self, soup: BeautifulSoup):
        for sel in self.DROP_SELECTORS:
            for el in soup.select(sel):
                el.decompose()

    def extract(self, html_text: str) -> dict:
        soup = BeautifulSoup(html_text or "", self.parser)
        self._strip_noise(soup)

        title = ""
        if soup.title:
            title = norm_space(soup.title.get_text(" "))

        # Try main selectors
        best = None
        for sel in self.MAIN_SELECTORS:
            cand = soup.select_one(sel)
            if cand:
                best = cand
                break

        # Fallback: choose the container with the most <p> text
        if not best:
            candidates = soup.find_all(["article", "section", "div"])
            best_len = 0
            for c in candidates:
                text = norm_space(c.get_text(" "))
                length = len(text)
                if length > best_len:
                    best_len = length
                    best = c

        text = ""
        if best:
            text = norm_space(best.get_text(" "))

        # Also collect top paragraphs as snippets
        paragraphs = []
        for p in soup.find_all("p"):
            t = norm_space(p.get_text(" "))
            if len(t) > 60:
                paragraphs.append(t)
            if len(paragraphs) >= 25:
                break

        # Collect links (absolute)
        links = set()
        for a in soup.find_all("a", href=True):
            links.add(a["href"])

        return {
            "title": title,
            "text": text,
            "paragraphs": paragraphs,
            "links": list(links),
        }

# ------------ Websurf (Main) ------------

class Websurf:
    """
    Websurf orchestrates: search -> fetch -> extract -> rank.
    Use `forward(query, ...)` to get everything needed for LLM context.
    """

    def __init__(
        self,
        max_pages: int = 20,
        n_results_per_engine: int = 8,
        per_domain_limit: int = 3,
        request_timeout: int = 15,
        rate_limit_sec: float = 0.5,
        user_agent: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.max_pages = max_pages
        self.n_results_per_engine = n_results_per_engine
        self.per_domain_limit = per_domain_limit
        self.rate_limit_sec = rate_limit_sec
        self.logger = logger or logging.getLogger("websurf")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.searcher = SearchAggregator(user_agent=user_agent)
        self.fetcher = PageFetcher(user_agent=user_agent, timeout=request_timeout)
        self.extractor = ContentExtractor()

    def _select_urls(self, search_results):
        # Enforce per-domain caps and max_pages
        counts = {}
        selected = []
        for r in search_results:
            u = clean_url(r.get("url", ""))
            if not u:
                continue
            d = domain(u)
            counts[d] = counts.get(d, 0) + 1
            if counts[d] <= self.per_domain_limit:
                selected.append(r)
            if len(selected) >= self.max_pages:
                break
        return selected

    def _fetch_and_extract(self, item, query_terms):
        u = item["url"]
        try:
            r = self.fetcher.get(u)
            ctype = r.headers.get("Content-Type", "")
            if is_probably_binary(ctype):
                return None

            # Handle obvious non-HTML content
            if "application/pdf" in (ctype or "").lower() or u.lower().endswith(".pdf"):
                # Optionally, you could integrate pdfminer.six to extract text.
                text = "[PDF detected; text extraction not enabled in this module]"
                ext = {"title": item.get("title") or "", "text": text, "paragraphs": [], "links": []}
            else:
                ext = self.extractor.extract(r.text)

            # Compute relevance
            combined_text = " ".join(
                [ext.get("title") or "", ext.get("text") or ""] + ext.get("paragraphs", [])[:10]
            )
            score = score_relevance(combined_text, query_terms)

            # Build summary snippet
            snippet = item.get("snippet") or ""
            if not snippet and ext.get("paragraphs"):
                snippet = truncate(ext["paragraphs"][0], 300)

            return {
                "url": u,
                "engine": item.get("engine", ""),
                "title": ext.get("title") or item.get("title") or "",
                "snippet": snippet,
                "relevance": score,
                "content": {
                    "text": ext.get("text", ""),
                    "paragraphs": ext.get("paragraphs", []),
                },
                "out_links": [clean_url(l) for l in ext.get("links", []) if clean_url(l)],
                "fetched_at": time.time(),
            }
        except Exception as e:
            self.logger.debug(f"Fetch failed: {u} ({e})")
            return None

    def forward(
        self,
        query: str,
        engines: list[str] | None = None,
        max_pages: int | None = None,
        per_domain_limit: int | None = None,
        n_results_per_engine: int | None = None,
        follow_out_links: bool = False,
        out_link_depth: int = 1,
        verbose: bool = True,
    ) -> dict:
        """
        Execute full pipeline:
          1) Search across engines for `query`
          2) Pick up to `max_pages` unique URLs (capped per-domain)
          3) Fetch + extract main content and key paragraphs
          4) (Optional) follow out-links up to `out_link_depth`
          5) Return a structured bundle ideal for LLM context

        Args:
            query: search query string
            engines: list of engine names to use (see SearchAggregator.search)
            max_pages: override global max_pages
            per_domain_limit: override per-domain cap
            n_results_per_engine: override #results per engine pre-dedup
            follow_out_links: if True, crawl discovered links too (breadth-first)
            out_link_depth: how many hops for out-link crawling

        Returns:
            {
                "query": ...,
                "stats": {...},
                "sources": [ {url, title, snippet, relevance, content{ text, paragraphs }, out_links, engine }, ... ],
                "links_all": [...],
                "summary": { "top_snippets": [...], "top_sources": [...] }
            }
        """
        t0 = time.time()
        if verbose:
            self.logger.info(f"Searching for: {query}")

        # Resolve overrides
        max_pages = max_pages or self.max_pages
        per_domain_limit = per_domain_limit or self.per_domain_limit
        n_results_per_engine = n_results_per_engine or self.n_results_per_engine

        # 1) Search
        search_results = self.searcher.search(
            query, n_per_engine=n_results_per_engine, use_engines=engines
        )
        if verbose:
            self.logger.info(f"Search results (pre-dedup): {len(search_results)}")

        # 2) Select initial URLs
        selected = self._select_urls(search_results)
        if verbose:
            self.logger.info(f"Selected URLs to visit: {len(selected)} (max_pages={max_pages}, per_domain={per_domain_limit})")

        # 3) Fetch + extract
        query_terms = tokenize(query)
        visited_urls = set()
        sources = []
        links_all = set()

        def process_batch(items):
            for item in items:
                u = clean_url(item["url"])
                if not u or u in visited_urls:
                    continue
                visited_urls.add(u)
                result = self._fetch_and_extract(item, query_terms)
                if result:
                    sources.append(result)
                    links_all.update(result.get("out_links", []))
                time.sleep(self.rate_limit_sec)

        # process initial batch
        process_batch(selected[:max_pages])

        # 4) Optional out-link crawling
        if follow_out_links and out_link_depth > 0 and len(sources) < max_pages:
            depth = 0
            frontier = []
            # seed with best sources' out-links
            for s in sorted(sources, key=lambda x: x.get("relevance", 0), reverse=True):
                for l in s.get("out_links", []):
                    frontier.append({"url": l, "title": "", "snippet": "", "engine": "outlink"})

            # BFS up to depth and max_pages
            while frontier and depth < out_link_depth and len(visited_urls) < max_pages:
                if verbose:
                    self.logger.info(f"Following out-links depth={depth+1} frontier={len(frontier)}")
                batch = []
                counts = {}
                for it in frontier:
                    if len(batch) + len(visited_urls) >= max_pages:
                        break
                    u = clean_url(it["url"])
                    if not u or u in visited_urls:
                        continue
                    d = domain(u)
                    counts[d] = counts.get(d, 0) + 1
                    if counts[d] <= per_domain_limit:
                        batch.append(it)
                frontier = []  # reset for next layer
                process_batch(batch)
                # collect next layer from newly added sources
                for s in sources:
                    for l in s.get("out_links", []):
                        if l not in visited_urls:
                            frontier.append({"url": l, "title": "", "snippet": "", "engine": "outlink"})
                depth += 1

        # 5) Rank & summarize
        sources.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        top_snips = []
        for s in sources[:10]:
            # prefer explicit snippet, else take first paragraph
            snip = s.get("snippet") or (s.get("content", {}).get("paragraphs")[:1] or [""])[0]
            top_snips.append(
                {
                    "url": s["url"],
                    "title": s.get("title", ""),
                    "snippet": truncate(snip, 400),
                }
            )

        elapsed = time.time() - t0
        result = {
            "query": query,
            "stats": {
                "elapsed_sec": round(elapsed, 3),
                "searched_results": len(search_results),
                "pages_fetched": len(sources),
                "unique_domains": len({domain(s["url"]) for s in sources}),
            },
            "sources": sources,  # full text & paragraphs included here
            "links_all": list(unique(links_all)),
            "summary": {
                "top_snippets": top_snips,
                "top_sources": [
                    {"url": s["url"], "title": s.get("title", ""), "relevance": s.get("relevance", 0.0)}
                    for s in sources[:10]
                ],
            },
        }
        return result


# ------------ CLI Usage ------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Websurf: multi-engine search + web content gatherer.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--max-pages", type=int, default=20, help="Max pages to fetch")
    parser.add_argument("--per-domain", type=int, default=3, help="Per-domain page cap")
    parser.add_argument("--n-per-engine", type=int, default=8, help="Results per engine prior to dedup")
    parser.add_argument("--follow-out-links", action="store_true", help="Follow discovered out-links")
    parser.add_argument("--out-link-depth", type=int, default=1, help="Depth for out-link crawling")
    parser.add_argument("--engines", type=str, default="", help="Comma-separated engines to use (default: auto)")
    parser.add_argument("--json", action="store_true", help="Print full JSON")
    args = parser.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()] or None

    ws = Websurf(
        max_pages=args.max_pages,
        per_domain_limit=args.per_domain,
        n_results_per_engine=args.n_per_engine,
    )
    out = ws.forward(
        query=args.query,
        engines=engines,
        follow_out_links=args.follow_out_links,
        out_link_depth=args.out_link_depth,
        verbose=True,
    )
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"# Query: {out['query']}")
        print(f"Elapsed: {out['stats']['elapsed_sec']}s | Fetched: {out['stats']['pages_fetched']} | Domains: {out['stats']['unique_domains']}")
        print("\n## Top Sources")
        for i, s in enumerate(out["summary"]["top_sources"], 1):
            print(f"{i:>2}. {s['title'] or '(no title)'}")
            print(f"    {s['url']}")
            print(f"    relevance={round(s['relevance'],2)}")
        print("\n## Snippets")
        for sn in out["summary"]["top_snippets"]:
            print(f"- {sn['title'] or sn['url']}: {sn['snippet']}")
