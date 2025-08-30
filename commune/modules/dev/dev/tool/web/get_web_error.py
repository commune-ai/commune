import re
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup   # pip install beautifulsoup4

# Error patterns you can tweak
DEFAULT_PATTERNS = [
    r"Failed to compile",
    r"Module not found:.*?Can't resolve\s+[\"'][^\"']+[\"']",
    r"Unhandled Runtime Error",
    r"Build error occurred",
    r"TypeError:\s.+",
    r"ReferenceError:\s.+",
    r"SyntaxError:\s.+",
    r"WebpackError",
    r"Next\.js.+Error",
    r"Cannot find module\s+[\"'][^\"']+[\"']",
    r"404\s+Not\s+Found",
    r"500\s+Internal\s+Server\s+Error",
]

@dataclass
class Hit:
    url: str
    pattern: str
    excerpt: str

class WebErrorProbe:
    """
    URL-only error probe (no server/container access).
    - GETs the URL (optionally retries)
    - Extracts visible text from typical error elements (pre/code/title/meta/div[role=alert])
    - Scans for common compile/runtime error signatures
    - forward(url) -> JSON-serializable dict
    """
    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        timeout: float = 5.0,
        retries: int = 2,
        user_agent: str = "Mozilla/5.0 (compatible; ErrorProbe/1.0)"
    ):
        self.patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in (patterns or DEFAULT_PATTERNS)]
        self.timeout = timeout
        self.retries = retries
        self.headers = {"User-Agent": user_agent, "Accept": "text/html, */*;q=0.8"}

    def _fetch(self, url: str) -> str:
        last_exc = None
        for _ in range(self.retries + 1):
            try:
                r = requests.get(url, timeout=self.timeout, headers=self.headers)
                # even on 4xx/5xx we still want the body to parse the error page
                return r.text or ""
            except Exception as e:
                last_exc = e
        return f"[HTTP ERROR] {type(last_exc).__name__}: {last_exc}"

    def _extract_text(self, html: str) -> str:
        """
        Pulls the most informative text from typical error surfaces.
        Works for Next.js dev overlay pages that render error text into <pre>/<code>,
        as well as generic server error pages.
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")

            chunks: List[str] = []

            # 1) Explicit error containers often used by dev overlays
            for sel in ["pre", "code", "[role=alert]", ".next-error", "#nextjs__container_errors"]:
                for node in soup.select(sel):
                    txt = node.get_text("\n", strip=False)
                    if txt and txt not in chunks:
                        chunks.append(txt)

            # 2) Fallback grab: the title (often contains “Failed to compile” or HTTP errors)
            title = soup.find("title")
            if title:
                chunks.append(title.get_text(" ", strip=True))

            # 3) Meta description sometimes carries error summary
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                chunks.append(meta_desc["content"])

            # 4) As a last resort, whole-page visible text (can be large)
            if not chunks:
                chunks.append(soup.get_text("\n", strip=False))

            text = "\n\n".join(chunks)
            return text
        except Exception:
            # If parsing fails, scan raw HTML
            return html

    def _scan(self, url: str, text: str) -> List[Hit]:
        hits: List[Hit] = []
        if not text:
            return hits
        for pat in self.patterns:
            for m in pat.finditer(text):
                start = max(m.start() - 180, 0)
                end = min(m.end() + 220, len(text))
                excerpt = text[start:end].strip()
                hits.append(Hit(url=url, pattern=pat.pattern, excerpt=excerpt))
        return hits

    def forward(self, url: str) -> Dict[str, Any]:
        """
        Run the probe against a single URL and return a structured report.
        """
        html = self._fetch(url)
        text = self._extract_text(html)
        matches = self._scan(url, text)

        return {
            "timestamp": int(time.time()),
            "target_url": url,
            "ok": len(matches) == 0,
            "matches": [
                {"url": h.url, "pattern": h.pattern, "excerpt": h.excerpt}
                for h in matches
            ],
        }

# ---------------- Example usage ----------------
if __name__ == "__main__":
    probe = WebErrorProbe()
    report = probe.forward("http://localhost:3000")
    print(json.dumps(report, indent=2))
