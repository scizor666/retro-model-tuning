import scrapy
from bs4 import BeautifulSoup
import jsonlines
import os

class HistoricalSpider(scrapy.Spider):
    name = "historical_spider"
    allowed_domains = ["nintendo.fandom.com"]
    start_urls = [
        "https://nintendo.fandom.com/wiki/Super_Nintendo_Entertainment_System",
        "https://nintendo.fandom.com/wiki/Nintendo_Entertainment_System",
        "https://nintendo.fandom.com/wiki/Nintendo_64",
        "https://nintendo.fandom.com/wiki/Game_Boy",
        "https://nintendo.fandom.com/wiki/List_of_Super_Nintendo_Entertainment_System_games",
        "https://nintendo.fandom.com/wiki/List_of_Nintendo_64_games"
    ]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1.0,
        'ROBOTSTXT_OBEY': False,
        'DEPTH_LIMIT': 1,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    def __init__(self, *args, **kwargs):
        super(HistoricalSpider, self).__init__(*args, **kwargs)
        self.output_file = 'data/historical_data.jsonl'
        os.makedirs('data', exist_ok=True)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'lxml')
        content = soup.find('div', class_='mw-parser-output')
        if not content:
            return

        title = response.css('h1.page-header__title::text').get()
        if not title:
            return

        # Clean extraneous elements
        for element in content.select('.navbox, .reference, .toc, script, style, table'):
            element.decompose()

        text = content.get_text(separator='\n')
        vlines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(vlines)

        if len(clean_text) > 200:
            with jsonlines.open(self.output_file, mode='a') as writer:
                writer.write({
                    'source': 'historical_wiki',
                    'url': response.url,
                    'title': title.strip(),
                    'text': clean_text
                })
