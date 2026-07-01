#!/usr/bin/env python3
"""Generate a ranked HuggingFace-downloads page for vLLM-Omni supported models.

Scrapes the authoritative model list from vllm-omni's docs, queries the public
HuggingFace API for each repo's download counts, and renders a static, sortable
HTML page plus a machine-readable data.json.

Stdlib only (no pip installs) so it runs on a bare GitHub Actions runner.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import html
import json
import os
import re
import urllib.error
import urllib.request

SUPPORTED_MD = (
    "https://raw.githubusercontent.com/vllm-project/vllm-omni/main/docs/models/supported_models.md"
)
OUT_DIR = os.environ.get("OUT_DIR", ".")
# Substrings that indicate a matched "org/name" token is NOT a HF model id.
_BAD = ("http", "github", ".md", ".py", "docs/", "vllm-omni", "vllm_omni", "/main", "/blob", "/tree", "main/recipes")


def get_model_ids() -> list[str]:
    req = urllib.request.Request(SUPPORTED_MD, headers={"User-Agent": "omni-rankings"})
    text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    ids: set[str] = set()
    # Prefer explicit huggingface.co links, then fall back to bare org/name tokens.
    ids.update(re.findall(r"huggingface\.co/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)", text))
    for tok in re.findall(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", text):
        low = tok.lower()
        if any(b in low for b in _BAD):
            continue
        ids.add(tok)
    return sorted(ids)


def _first_public_commit(m: str) -> str | None:
    """Oldest *visible* commit date (YYYY-MM-DD) as a 'went public' proxy.

    For normal repos this equals the HF ``createdAt`` (the initial commit).
    For repos that were staged private then squashed at release (e.g. a
    ``Super-squash branch 'main'`` commit, as NVIDIA did for Cosmos3-Nano),
    the private history is discarded, so the oldest visible commit is the
    publish date rather than the private-staging date. Returns None if the
    commits API is unavailable (gated repos 401 here)."""
    url = f"https://huggingface.co/api/models/{m}/commits/main?limit=1000"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "omni-rankings"})
        with urllib.request.urlopen(req, timeout=25) as r:
            commits = json.load(r)
        if commits:
            return (commits[-1].get("date") or "")[:10] or None
    except Exception:  # noqa: BLE001
        return None
    return None


def fetch(m: str) -> dict:
    url = f"https://huggingface.co/api/models/{m}?expand[]=downloads&expand[]=downloadsAllTime&expand[]=likes&expand[]=pipeline_tag&expand[]=createdAt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "omni-rankings"})
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.load(r)
        created = (d.get("createdAt") or "")[:10]  # HF repo-creation date
        released = _first_public_commit(m) or created or None  # went-public proxy
        return {
            "id": m,
            "downloads_30d": d.get("downloads"),
            "downloads_all": d.get("downloadsAllTime"),
            "likes": d.get("likes"),
            "pipeline_tag": d.get("pipeline_tag"),
            "created": created or None,
            "released": released,
            "status": "ok",
        }
    except urllib.error.HTTPError as e:
        return {"id": m, "downloads_30d": None, "downloads_all": None, "likes": None,
                "pipeline_tag": None, "created": None, "released": None,
                "status": ("gated" if e.code == 401 else f"http_{e.code}")}
    except Exception as e:  # noqa: BLE001
        return {"id": m, "downloads_30d": None, "downloads_all": None, "likes": None,
                "pipeline_tag": None, "created": None, "released": None, "status": type(e).__name__}


def collect() -> list[dict]:
    models = get_model_ids()
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(fetch, models))
    rows.sort(key=lambda r: (r["downloads_all"] is None, -(r["downloads_all"] or 0)))
    return rows


def render_html(rows: list[dict], generated: str) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    t30 = sum(r["downloads_30d"] or 0 for r in ok)
    tall = sum(r["downloads_all"] or 0 for r in ok)

    def cell(n):
        return f"{n:,}" if isinstance(n, int) else "&mdash;"

    trs = []
    for i, r in enumerate(rows, 1):
        mid = html.escape(r["id"])
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "")
        tag = html.escape(r["pipeline_tag"] or "")
        note = "" if r["status"] == "ok" else f' <span class="badge">{html.escape(r["status"])}</span>'
        released = r.get("released") or ""
        trs.append(
            f'<tr>'
            f'<td class="rank">{medal or i}</td>'
            f'<td class="model"><a href="https://huggingface.co/{mid}" target="_blank" rel="noopener">{mid}</a>{note}</td>'
            f'<td class="tag">{tag}</td>'
            f'<td class="date" data-v="{released}">{released or "&mdash;"}</td>'
            f'<td class="num" data-v="{r["downloads_30d"] or 0}">{cell(r["downloads_30d"])}</td>'
            f'<td class="num" data-v="{r["downloads_all"] or 0}">{cell(r["downloads_all"])}</td>'
            f'<td class="num" data-v="{r["likes"] or 0}">{cell(r["likes"])}</td>'
            f'</tr>'
        )
    rows_html = "\n".join(trs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM-Omni Rankings &mdash; HuggingFace Downloads</title>
<style>
  :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#30363d; --accent:#4493f8; --card:#161b22; }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --accent:#0969da; --card:#f6f8fa; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--fg); }}
  .wrap {{ max-width:960px; margin:0 auto; padding:32px 20px 80px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); margin:0 0 20px; font-size:14px; }}
  .sub a {{ color:var(--accent); }}
  .bar {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:16px; }}
  input[type=search] {{ flex:1; min-width:200px; padding:8px 12px; border:1px solid var(--line);
        border-radius:8px; background:var(--card); color:var(--fg); font-size:14px; }}
  .stat {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px; font-size:13px; }}
  .stat b {{ color:var(--accent); }}
  table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }}
  th {{ position:sticky; top:0; background:var(--bg); cursor:pointer; user-select:none; font-size:13px; color:var(--muted); white-space:nowrap; }}
  th.num, td.num {{ text-align:right; }}
  th:hover {{ color:var(--fg); }}
  th[data-sorted]::after {{ content:" ▲"; }}
  th[data-sorted="desc"]::after {{ content:" ▼"; }}
  td.rank {{ color:var(--muted); width:44px; text-align:center; }}
  td.model a {{ color:var(--accent); text-decoration:none; }}
  td.model a:hover {{ text-decoration:underline; }}
  td.tag {{ color:var(--muted); font-size:12px; }}
  td.date {{ color:var(--muted); font-size:12px; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .badge {{ font-size:11px; color:#f0883e; border:1px solid #f0883e55; border-radius:6px; padding:0 5px; }}
  footer {{ margin-top:24px; color:var(--muted); font-size:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>vLLM-Omni Rankings</h1>
  <p class="sub">Ranked by HuggingFace downloads &middot; auto-generated from
    <a href="https://github.com/vllm-project/vllm-omni/blob/main/docs/models/supported_models.md" target="_blank" rel="noopener">supported_models.md</a>
    &middot; updated every 6h</p>
  <div class="bar">
    <input id="q" type="search" placeholder="Filter models&hellip;">
    <span class="stat">{len(ok)} models</span>
    <span class="stat">30d: <b>{t30:,}</b></span>
    <span class="stat">all-time: <b>{tall:,}</b></span>
  </div>
  <table id="t">
    <thead><tr>
      <th class="rank">#</th>
      <th data-key="model">Model</th>
      <th data-key="tag">Task</th>
      <th data-key="date">Released</th>
      <th class="num" data-key="num">30-day</th>
      <th class="num" data-key="num" data-sorted="desc">All-time</th>
      <th class="num" data-key="num">Likes</th>
    </tr></thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
  <footer>
    Last updated {generated}. Data: HuggingFace public API
    (<code>downloads</code> = last 30 days, <code>downloadsAllTime</code> = cumulative).
    <b>Released</b> = oldest visible commit on the HF repo (a "went public" proxy;
    for gated repos it falls back to the repo-creation date).
    Some entries are the upstream base weights an integration builds on.
    Machine-readable: <a href="data.json">data.json</a>.
  </footer>
</div>
<script>
const tbody = document.querySelector('#t tbody');
const rows = [...tbody.rows];
document.querySelectorAll('#t th[data-key]').forEach((th, idx) => {{
  th.addEventListener('click', () => {{
    const col = th.cellIndex;
    const isNum = th.classList.contains('num');
    const desc = th.getAttribute('data-sorted') !== 'desc';
    document.querySelectorAll('#t th').forEach(h => h.removeAttribute('data-sorted'));
    th.setAttribute('data-sorted', desc ? 'desc' : 'asc');
    const sorted = rows.slice().sort((a, b) => {{
      let x, y;
      if (isNum) {{ x = +a.cells[col].dataset.v; y = +b.cells[col].dataset.v; }}
      else {{ x = a.cells[col].innerText.toLowerCase(); y = b.cells[col].innerText.toLowerCase(); }}
      return (x < y ? -1 : x > y ? 1 : 0) * (desc ? -1 : 1);
    }});
    sorted.forEach((r, i) => {{ r.cells[0].innerText = i + 1 < 4 ? ['🥇','🥈','🥉'][i] : (i + 1); tbody.appendChild(r); }});
  }});
}});
document.querySelector('#q').addEventListener('input', e => {{
  const v = e.target.value.toLowerCase();
  rows.forEach(r => {{ r.style.display = r.cells[1].innerText.toLowerCase().includes(v) ? '' : 'none'; }});
}});
</script>
</body>
</html>
"""


def main() -> None:
    rows = collect()
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "data.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": generated, "source": SUPPORTED_MD, "models": rows}, f, indent=2)
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_html(rows, generated))
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"generated {OUT_DIR}/index.html + data.json: {ok}/{len(rows)} models ok @ {generated}")


if __name__ == "__main__":
    main()
