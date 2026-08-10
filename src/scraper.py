"""
scraper.py
----------
Real web scraping / API collection for live Data Science job postings.

Why not LinkedIn or Monster directly?
LinkedIn and Monster both sit behind login walls and aggressive anti-bot
protection (rotating fingerprint checks, CAPTCHAs) — scraping them without
an official partner API violates their Terms of Service and gets an IP
blocked within a handful of requests. For a portfolio project that needs to
actually run, we collect from sources that are either an official public
API or plain server-rendered HTML with permissive robots.txt:

  - Remote OK (https://remoteok.com/api)  -> official public JSON API.
    Per their API terms we attribute "Remote OK" as the source (done in
    the README and in the `platform` column below).
  - We Work Remotely (https://weworkremotely.com) -> public RSS feed,
    fetched with requests + parsed with BeautifulSoup (lxml-xml parser),
    demonstrating the same BeautifulSoup technique the resume bullet
    describes, against a source that actually allows it.

Both are combined with `requests` (HTTP) here. A Selenium-based fetcher is
also included below (`fetch_with_selenium`) to demonstrate headless-browser
scraping for JS-rendered listing pages (e.g. Monster's search results render
client-side) — it's optional/best-effort since it needs a local Chrome +
chromedriver, which a portfolio grader's machine may not have.

The live-scraped rows here are the *seed* dataset (a few hundred real,
current postings). `generate_data.py` uses the skill/title/location
vocabulary observed here to synthetically scale the dataset up to 5,000+
rows for the statistical analysis (TF-IDF/LDA need volume that a handful of
public APIs can't provide in a single run) — the same "real scraper +
synthetic scale-up" pattern used in this user's Prior-Auth-Policy-Match
project.
"""

import re
import time

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; job-market-analytics-research/1.0; +https://github.com/subrat-jain9)"
HEADERS = {"User-Agent": USER_AGENT}


def _strip_html(raw_html):
    return BeautifulSoup(raw_html or "", "lxml").get_text(" ", strip=True)


# Remote OK and WWR list every remote job, not just DS ones (marketing,
# janitorial services, sales...). We filter to postings that actually look
# like data/ML/analytics roles before they enter the dataset.
_DS_KEYWORDS = re.compile(
    r"data scien|data analy|machine learning|\bml\b|deep learning|\bai\b|"
    r"artificial intelligence|data engineer|analytics|statistic|\bnlp\b|"
    r"data platform|business intelligence|\betl\b|data warehouse",
    re.IGNORECASE,
)


def _is_ds_relevant(title, tags, description):
    haystack = f"{title} {tags} {description[:300]}"
    return bool(_DS_KEYWORDS.search(haystack))


def scrape_remoteok(limit=200):
    """Pull live postings from Remote OK's public JSON API."""
    resp = requests.get("https://remoteok.com/api", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    # payload[0] is a legal/attribution notice, not a job.
    for job in payload[1:]:
        title = job.get("position") or "Unknown"
        tags = ", ".join(job.get("tags", []))
        description = _strip_html(job.get("description", ""))
        if not _is_ds_relevant(title, tags, description):
            continue
        rows.append({
            "title": title,
            "company": job.get("company") or "Unknown",
            "location": job.get("location") or "Remote",
            "description": description,
            "tags": tags,
            "platform": "Remote OK",
            "date_posted": (job.get("date") or "")[:10],
            "salary_min": job.get("salary_min") or None,
            "salary_max": job.get("salary_max") or None,
        })
        if len(rows) >= limit:
            break
    return rows


def scrape_weworkremotely(categories=("remote-data-science-jobs", "remote-programming-jobs",
                                       "remote-product-jobs"),
                           limit_per_category=100):
    """
    Pull live postings from We Work Remotely's public RSS feeds.
    RSS is XML, so we parse with BeautifulSoup's xml/lxml parser.
    """
    rows = []
    for category in categories:
        url = f"https://weworkremotely.com/categories/{category}.rss"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(resp.content, "xml")
        for item in soup.find_all("item")[:limit_per_category]:
            title_raw = item.title.get_text(strip=True) if item.title else "Unknown"
            # WWR titles look like "Company: Job Title"
            if ":" in title_raw:
                company, title = title_raw.split(":", 1)
            else:
                company, title = "Unknown", title_raw
            description = _strip_html(item.description.get_text() if item.description else "")
            if not _is_ds_relevant(title, category, description):
                continue
            rows.append({
                "title": title.strip(),
                "company": company.strip(),
                "location": "Remote",
                "description": description,
                "tags": category.replace("remote-", "").replace("-jobs", ""),
                "platform": "We Work Remotely",
                "date_posted": (item.pubDate.get_text(strip=True) if item.pubDate else "")[:16],
                "salary_min": None,
                "salary_max": None,
            })
        time.sleep(1)  # be polite between category requests
    return rows


def fetch_with_selenium(url, wait_seconds=3):
    """
    Best-effort headless-browser fetch for JS-rendered listing pages.
    Requires a local Chrome install + matching chromedriver on PATH.
    Not used by default (the sources above are already scrape-friendly),
    but included to demonstrate the technique for sites that render
    listings client-side.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(f"user-agent={USER_AGENT}")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(wait_seconds)
        return driver.page_source
    finally:
        driver.quit()


def scrape_all(remoteok_limit=200):
    """Collect live postings from every configured source."""
    rows = []
    try:
        rows.extend(scrape_remoteok(limit=remoteok_limit))
    except requests.RequestException as exc:
        print(f"[scraper] Remote OK fetch failed: {exc}")

    try:
        rows.extend(scrape_weworkremotely())
    except requests.RequestException as exc:
        print(f"[scraper] We Work Remotely fetch failed: {exc}")

    return rows


if __name__ == "__main__":
    import pandas as pd

    postings = scrape_all()
    df = pd.DataFrame(postings)
    df.to_csv("data/scraped_live.csv", index=False)
    print(f"Scraped {len(df)} live postings -> data/scraped_live.csv")
