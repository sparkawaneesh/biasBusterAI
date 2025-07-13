import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
from typing import Dict, Optional

class URLScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.timeout = 10
        
    def scrape_article(self, url: str) -> Dict[str, any]:
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL provided")
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            author = self._extract_author(soup)
            publish_date = self._extract_publish_date(soup)
            
            clean_content = self._clean_content(content)
            
            if len(clean_content) < 100:
                raise ValueError("Insufficient content extracted from URL")
            
            return {
                'title': title,
                'content': clean_content,
                'author': author,
                'publish_date': publish_date,
                'url': url,
                'domain': parsed_url.netloc,
                'word_count': len(clean_content.split())
            }
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to scrape content: {str(e)}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        # Try multiple selectors
        title_selectors = [
            'h1',
            'title',
            '[class*="title"]',
            '[class*="headline"]',
            '[data-testid*="title"]',
            '.entry-title',
            '.post-title',
            '.article-title',
            '.headline',
            'h1.title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element and element.get_text().strip():
                return element.get_text().strip()
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        return "Untitled Article"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'aside', '.sidebar', '.menu', '.advertisement', 
                           '.ad', '.comments', '.comment', '.social-share']):
            element.decompose()
        
        # Try multiple content selectors
        content_selectors = [
            'article',
            '[class*="content"]',
            '[class*="article"]',
            '[class*="post"]',
            '[class*="entry"]',
            '[class*="story"]',
            '[data-testid*="content"]',
            '.main-content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.article-body',
            '.story-body',
            'main',
            '.content'
        ]
        
        best_content = ""
        max_length = 0
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if len(text) > max_length:
                    max_length = len(text)
                    best_content = text
        
        # Fallback: extract from paragraphs
        if len(best_content) < 200:
            paragraphs = soup.find_all('p')
            paragraph_texts = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            best_content = ' '.join(paragraph_texts)
        
        return best_content
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article author"""
        author_selectors = [
            '[class*="author"]',
            '[class*="byline"]',
            '[data-testid*="author"]',
            '.author-name',
            '.byline',
            '[rel="author"]',
            '[itemprop="author"]',
            '.writer',
            '.journalist'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element and element.get_text().strip():
                return element.get_text().strip()
        
        return None
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publish date"""
        date_selectors = [
            '[class*="date"]',
            '[class*="published"]',
            '[class*="time"]',
            '[data-testid*="date"]',
            '.publish-date',
            '.post-date',
            '[itemprop="datePublished"]',
            'time',
            '.timestamp'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                # Try to get datetime attribute first
                date_text = (element.get('datetime') or 
                           element.get('content') or 
                           element.get_text().strip())
                if date_text:
                    return date_text
        
        return None
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might interfere with analysis
        content = re.sub(r'[^\w\s\.,!?;:\-\'\"()]', ' ', content)
        
        # Remove very long strings of repeated characters
        content = re.sub(r'(.)\1{10,}', r'\1', content)
        
        # Trim and return
        return content.strip()
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False
    
    def get_domain_info(self, url: str) -> Dict[str, str]:
        """Get basic domain information"""
        try:
            parsed = urlparse(url)
            return {
                'domain': parsed.netloc,
                'scheme': parsed.scheme,
                'path': parsed.path
            }
        except:
            return {'domain': 'unknown', 'scheme': 'unknown', 'path': ''}

if __name__ == "__main__":
    # Test the scraper
    scraper = URLScraper()
    
    test_urls = [
        "https://example.com/article",  # This will fail but shows error handling
    ]
    
    for url in test_urls:
        try:
            result = scraper.scrape_article(url)
            print(f"✅ Successfully scraped: {result['title']}")
            print(f"📝 Content length: {len(result['content'])} characters")
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")