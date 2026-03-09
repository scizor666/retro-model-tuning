import scrapy
from bs4 import BeautifulSoup
import jsonlines
import os
import hashlib
import requests
from urllib.parse import urljoin
import re

class MultimodalSpider(scrapy.Spider):
    name = "multimodal_images"
    allowed_domains = ["emulation.gametechwiki.com"]
    start_urls = ["https://emulation.gametechwiki.com/index.php/Category:Emulators"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1.0,
        'ROBOTSTXT_OBEY': False,
        'DEPTH_LIMIT': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'USER_AGENT': 'Mozilla/5.0'
    }

    def __init__(self, *args, **kwargs):
        super(MultimodalSpider, self).__init__(*args, **kwargs)
        self.output_file = 'data/multimodal_dataset.jsonl'
        self.image_dir = 'data/images'
        
        os.makedirs(self.image_dir, exist_ok=True)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def parse(self, response):
        # Find emulator pages
        for href in response.css('a::attr(href)').getall():
            if href.startswith('/index.php/') and ':' not in href.replace('Category:',''):
                yield response.follow(href, self.parse_page)
                
    def parse_page(self, response):
        soup = BeautifulSoup(response.text, 'lxml')
        content = soup.find('div', id='mw-content-text')
        if not content:
            return
            
        title = response.css('h1#firstHeading::text').get()
        if not title:
            return
            
        images_processed = 0
        
        # Look for images, especially in infoboxes or thumbnails
        for img in content.find_all('img'):
            src = img.get('src')
            if not src:
                continue
                
            img_url = urljoin(response.url, src)
            
            # Skip UI elements
            if any(x in img_url.lower() for x in ['button', 'icon', 'logo.png', 'ui']):
                continue
                
            # Heuristic for finding a caption
            caption = img.get('alt', '').strip()
            
            # Often images are inside a 'thumbinner' div with a 'thumbcaption'
            thumbinner = img.find_parent('div', class_='thumbinner')
            if thumbinner:
                caption_elem = thumbinner.find('div', class_='thumbcaption')
                if caption_elem:
                    caption = caption_elem.get_text(strip=True)
            
            # If no good caption, use the page title as context
            if len(caption) < 10:
                caption = f"Screenshot or logo related to {title} emulator."

            # Download it
            try:
                img_response = requests.get(img_url, timeout=10)
                if img_response.status_code == 200:
                    # Let's verify it's an actual image by size (skip tiny ones < 5KB)
                    if len(img_response.content) > 5000:
                        filename = hashlib.md5(img_url.encode()).hexdigest() + ".jpg"
                        filepath = os.path.join(self.image_dir, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(img_response.content)
                            
                        with jsonlines.open(self.output_file, mode='a') as writer:
                            writer.write({
                                "id": filename.split('.')[0],
                                "image": filepath,
                                "conversations": [
                                    {
                                        "from": "human",
                                        "value": "<image>\nWhat is this?"
                                    },
                                    {
                                        "from": "gpt",
                                        "value": f"This is an image showing: {caption}"
                                    }
                                ],
                                "source": response.url
                            })
                        images_processed += 1
            except Exception as e:
                self.logger.error(f"Failed to process {img_url}: {e}")
                
        self.logger.info(f"Processed {images_processed} multimodal pairs from {title}")
