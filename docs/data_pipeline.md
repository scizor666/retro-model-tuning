# Data Pipeline

High-quality responses from an LLM require high-quality data. Because retro gaming troubleshooting information is scattered across obscure forums, wikis, and documentation, we had to build custom crawlers.

## 1. Web Crawling (`crawlers/`)

We targeted three distinct types of data sources using Python, saving output as `.jsonl` (JSON Lines) to handle large volumes of text seamlessly without loading everything into memory.

### A. Emulation General Wiki (`emuwiki_spider.py`)
Uses the `scrapy` framework.
- **Why:** The most comprehensive source of emulator comparisons, setup gotchas, and graphical glitches.
- **Challenge:** The wiki actively blocks default bot user-agents (`HTTP 403 Forbidden`). We bypassed this by providing a standard Chrome `USER_AGENT` in the Scrapy settings.
- **Mechanic:** Recursively finds links matching `/index.php/`, stripping out special pages (`Talk:`, `Category:`), navboxes, and table-of-contents elements via `BeautifulSoup` to leave only pure informational text.

### B. Libretro Docs (`libretro_spider.py`)
Uses the `scrapy` framework.
- **Why:** The official documentation config for RetroArch and its cores.
- **Challenge:** The docs are highly structured Markdown-to-HTML conversions.
- **Mechanic:** We specifically target the left-side navigation links (`nav.md-nav a`) and extract the inner article content (`.md-content__inner`), throwing away the layout wrappers.

### C. Reddit Troubleshooting (`reddit_scraper.py`)
Uses the `requests` library.
- **Why:** Wikis tell you how to set things up; Reddit tells you how things break. This is vital for Q&A troubleshooting.
- **Challenge:** Official Reddit API requires OAuth, but public searches can still be accessed via `.json` endpoints.
- **Mechanic:** Searches top subreddits (`r/emulation`, `r/snes`, etc.) for keywords like "issue", "patch", and "help". We apply a minimum length filter to ensure the thread body has enough substance to formulate an answer.

## 2. Text Chunking (`processing/chunker.py`)

Language models have finite context windows. Feeding an entire 5,000-word wiki page to a generator model at once dilutes its attention and yields poor synthetic Q&A.

We use `RecursiveCharacterTextSplitter` from LangChain to break the raw text into manageable pieces:
- **Chunk Size:** 1000 characters.
- **Overlap:** 150 characters. (Overlap ensures that if a sentence spans the boundary of a chunk, the context isn't lost).

For every chunk, we heavily prepend the `Title:` of the article so that the generator always knows *what* emulator or game the isolated chunk of text is actually talking about.
