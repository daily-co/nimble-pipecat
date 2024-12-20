import requests
from bs4 import BeautifulSoup
import time
import os
from pathlib import Path
import logging
from urllib.parse import urljoin
import html2text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlogScraper:
    def __init__(self, base_url, output_dir="blog_posts"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Custom Bot - Educational Purpose",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            }
        )
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = False
        self.converter.body_width = 0

    def get_page(self, url):
        try:
            time.sleep(2)
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def parse_blog_posts(self, html_content):
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        posts = soup.find_all("article", class_="index-item")

        blog_data = []
        for post in posts:
            try:
                title_container = post.find("h3", class_="index-item-text-title")
                if not title_container:
                    continue

                title_link = title_container.find("a")
                if not title_link:
                    continue

                title = title_link.get_text(strip=True)
                link = title_link["href"]

                logger.info(f"Found post: {title}")
                blog_data.append({"title": title, "url": urljoin(self.base_url, link)})
            except Exception as e:
                logger.warning(f"Error parsing post: {str(e)}")
                continue

        return blog_data

    def convert_to_plain_text(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        output_text = ""

        # Find and add the article title if it exists
        article_title = soup.find("h1", class_="article-title")
        if article_title:
            title_text = article_title.get_text(strip=True)
            output_text += f"# {title_text}\n\n"

        # Find the entry-content div
        content_div = soup.find("div", class_="entry-content")
        if content_div:
            output_text += self.converter.handle(str(content_div))
            return output_text

        logger.warning(
            "Could not find entry-content div, falling back to full page content"
        )
        return output_text + self.converter.handle(html_content)

    def save_post(self, title, content):
        safe_title = "".join(
            c for c in title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        filename = self.output_dir / f"{safe_title}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Saved: {filename}")
        except IOError as e:
            logger.error(f"Error saving {filename}: {e}")

    def process_blog_posts(self):
        logger.info("Starting blog post processing")

        html_content = self.get_page(self.base_url)
        if not html_content:
            return

        blog_posts = self.parse_blog_posts(html_content)
        logger.info(f"Found {len(blog_posts)} posts")

        for post in blog_posts:
            logger.info(f"Processing: {post['title']}")
            post_html = self.get_page(post["url"])
            if post_html:
                plain_text = self.convert_to_plain_text(post_html)
                self.save_post(post["title"], plain_text)


def main():
    base_url = "https://nvidianews.nvidia.com/news/class=?q=agents&year="
    try:
        scraper = BlogScraper(base_url)
        scraper.process_blog_posts()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
