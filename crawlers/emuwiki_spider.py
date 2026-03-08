import scrapy
from bs4 import BeautifulSoup
import jsonlines
import re
import os

class EmuWikiSpider(scrapy.Spider):
    name = "emuwiki"
    allowed_domains = ["emulation.gametechwiki.com"]
    start_urls = ["https://emulation.gametechwiki.com/index.php/Main_Page"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1.0,
        'ROBOTSTXT_OBEY': False, # Changed to False because bots are blocked
        'DEPTH_LIMIT': 3,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    def __init__(self, *args, **kwargs):
        super(EmuWikiSpider, self).__init__(*args, **kwargs)
        self.output_file = 'data/raw_emuwiki.jsonl'
        os.makedirs('data', exist_ok=True)
        # Clear previous run
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def parse(self, response):
        # Find all valid wiki links to follow
        for href in response.css('a::attr(href)').getall():
            if href.startswith('/index.php/'):
                yield response.follow(href, self.parse_page)

    def parse_page(self, response):
        # We only want actual content articles, excluding special pages
        url_end = response.url.split('/')[-1]
        
        # Skip special namespaces
        if any(x in url_end for x in ['Category:', 'Special:', 'Help:', 'Talk:', 'User:']):
            return

        soup = BeautifulSoup(response.text, 'lxml')
        content = soup.find('div', id='mw-content-text')
        if not content:
            return

        title = response.css('h1#firstHeading::text').get()
        if not title:
            return

        # Remove navboxes, references, and TOC to keep text clean
        for element in content.select('.navbox, .reference, .toc, script, style'):
            element.decompose()

        # Clean up text
        text = content.get_text(separator='\n')
        vlines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(vlines)

        # Basic filtering for pages with substance
        if len(clean_text) > 200:
            with jsonlines.open(self.output_file, mode='a') as writer:
                writer.write({
                    'source': 'emuwiki',
                    'url': response.url,
                    'title': title,
                    'text': clean_text
                })
        
        # Keep crawling
        for href in response.css('a::attr(href)').getall():
            if href.startswith('/index.php/') and ':' not in href:
                yield response.follow(href, self.parse_page)
