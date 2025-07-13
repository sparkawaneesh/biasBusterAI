import axios from 'axios';
import * as cheerio from 'cheerio';
import { parse } from 'node-html-parser';

export interface ScrapedContent {
  title: string;
  content: string;
  url: string;
  wordCount: number;
  author?: string;
  publishDate?: string;
  domain: string;
}

export class URLScraper {
  private readonly userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36';
  private readonly timeout = 10000; // 10 seconds

  async scrapeArticle(url: string): Promise<ScrapedContent> {
    try {
      const response = await axios.get(url, {
        headers: {
          'User-Agent': this.userAgent,
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.5',
          'Accept-Encoding': 'gzip, deflate',
          'Connection': 'keep-alive',
          'Upgrade-Insecure-Requests': '1',
        },
        timeout: this.timeout,
        maxRedirects: 5,
      });

      const html = response.data;
      const $ = cheerio.load(html);
      const root = parse(html);

      // Extract title
      const title = this.extractTitle($);
      
      // Extract main content
      const content = this.extractContent($);
      
      // Extract metadata
      const author = this.extractAuthor($);
      const publishDate = this.extractPublishDate($);
      const domain = new URL(url).hostname;

      // Clean and validate content
      const cleanContent = this.cleanContent(content);
      const wordCount = cleanContent.split(/\s+/).length;

      if (!cleanContent || cleanContent.length < 100) {
        throw new Error('Insufficient content extracted from URL');
      }

      return {
        title,
        content: cleanContent,
        url,
        wordCount,
        author,
        publishDate,
        domain,
      };
    } catch (error) {
      console.error('Error scraping URL:', error);
      throw new Error(`Failed to scrape content from URL: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private extractTitle($: cheerio.CheerioAPI): string {
    // Try multiple selectors for title
    const titleSelectors = [
      'h1',
      'title',
      '[class*="title"]',
      '[class*="headline"]',
      '[data-testid*="title"]',
      '.entry-title',
      '.post-title',
      '.article-title',
    ];

    for (const selector of titleSelectors) {
      const element = $(selector).first();
      if (element.length && element.text().trim()) {
        return element.text().trim();
      }
    }

    return 'Untitled Article';
  }

  private extractContent($: cheerio.CheerioAPI): string {
    // Remove unwanted elements
    $('script, style, nav, header, footer, aside, .sidebar, .menu, .advertisement, .ad, .comments').remove();

    // Try multiple selectors for main content
    const contentSelectors = [
      'article',
      '[class*="content"]',
      '[class*="article"]',
      '[class*="post"]',
      '[class*="entry"]',
      '[data-testid*="content"]',
      '.main-content',
      '.post-content',
      '.entry-content',
      '.article-content',
      'main',
      '.content',
    ];

    for (const selector of contentSelectors) {
      const element = $(selector).first();
      if (element.length) {
        const text = element.text().trim();
        if (text.length > 200) {
          return text;
        }
      }
    }

    // Fallback: extract from paragraphs
    const paragraphs = $('p').map((_, el) => $(el).text().trim()).get();
    return paragraphs.join(' ');
  }

  private extractAuthor($: cheerio.CheerioAPI): string | undefined {
    const authorSelectors = [
      '[class*="author"]',
      '[class*="byline"]',
      '[data-testid*="author"]',
      '.author-name',
      '.byline',
      '[rel="author"]',
      '[itemprop="author"]',
    ];

    for (const selector of authorSelectors) {
      const element = $(selector).first();
      if (element.length && element.text().trim()) {
        return element.text().trim();
      }
    }

    return undefined;
  }

  private extractPublishDate($: cheerio.CheerioAPI): string | undefined {
    const dateSelectors = [
      '[class*="date"]',
      '[class*="published"]',
      '[data-testid*="date"]',
      '.publish-date',
      '.post-date',
      '[itemprop="datePublished"]',
      'time',
    ];

    for (const selector of dateSelectors) {
      const element = $(selector).first();
      if (element.length) {
        const dateText = element.text().trim() || element.attr('datetime') || element.attr('content');
        if (dateText) {
          return dateText;
        }
      }
    }

    return undefined;
  }

  private cleanContent(content: string): string {
    return content
      .replace(/\s+/g, ' ')
      .replace(/\n+/g, ' ')
      .replace(/\t+/g, ' ')
      .trim();
  }
}

export const urlScraper = new URLScraper();