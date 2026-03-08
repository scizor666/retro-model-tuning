import scrapy
from bs4 import BeautifulSoup
import jsonlines
import os

class LibretroDocsSpider(scrapy.Spider):
    name = "libretro"
    allowed_domains = ["docs.libretro.com"]
    start_urls = ["https://docs.libretro.com/"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 0.5,
        'ROBOTSTXT_OBEY': True,
        'DEPTH_LIMIT': 3,
    }

    def __init__(self, *args, **kwargs):
        super(LibretroDocsSpider, self).__init__(*args, **kwargs)
        self.output_file = 'data/raw_libretro.jsonl'
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def parse(self, response):
        # Follow left nav links
        for href in response.css('nav.md-nav a.md-nav__link::attr(href)').getall():
            yield response.follow(href, self.parse_page)

    def parse_page(self, response):
        soup = BeautifulSoup(response.text, 'lxml')
        content = soup.find('article', class_='md-content__inner')
        if not content:
            return

        title = response.css('h1::text').get()
        if not title:
            return

        # Clean up text
        text = content.get_text(separator='\n')
        vlines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(vlines)

        if len(clean_text) > 100:
            with jsonlines.open(self.output_file, mode='a') as writer:
                writer.write({
                    'source': 'libretro_docs',
                    'url': response.url,
                    'title': title.strip(),
                    'text': clean_text
                })
        
        # Follow next page links
        next_page = response.css('a.md-footer-nav__link--next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse_page)
