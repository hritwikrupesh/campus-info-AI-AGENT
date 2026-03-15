import os
import time
import json
import logging
import urllib.parse
from collections import deque
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
START_URL = "https://www.anits.info/"
ALLOWED_DOMAIN = "www.anits.info"
MAX_PAGES = 200
MAX_DEPTH = 3
SAVE_DIR = "data/raw"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Ensure the raw data directory exists."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        logger.info(f"Created directory: {SAVE_DIR}")


def get_slug_from_url(url: str) -> str:
    """Generate a safe filename slug from the URL."""
    parsed = urllib.parse.urlparse(url)
    # Remove leading and trailing slashes, replace intermediate with hyphens
    path = parsed.path.strip("/").replace("/", "-")
    
    if not path or path == "":
        return "index"
    
    # Clean up characters not safe for filenames
    safe_path = "".join(c if c.isalnum() or c in "-" else "_" for c in path)
    return safe_path


def fetch_page(url: str) -> str:
    """
    Fetch the HTML content of the page safely.
    Handles timeouts and connection errors gracefully.
    """
    try:
        # User-Agent to prevent getting blocked by basic bot protections
        headers = {"User-Agent": "SmartCampusAI/1.0 (Educational Bot)"}
        response = requests.get(url, headers=headers, timeout=10)
        
        # Raise an exception for HTTP error codes
        response.raise_for_status()
        
        # Check if the response is actually HTML
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            logger.warning(f"Skipping non-HTML page: {url} ({content_type})")
            return None
            
        return response.text
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout occurred fetching: {url}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error fetching: {url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        
    return None


def extract_links(html: str, base_url: str) -> list:
    """
    Extract internal links from the HTML page that belong to the allowed domain.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        
        # Convert relative URLs to absolute URLs
        absolute_url = urllib.parse.urljoin(base_url, href)
        
        # Remove URL fragments (everything after #) to prevent duplicate crawling
        absolute_url = urllib.parse.urldefrag(absolute_url)[0]
        
        # Ensure the link belongs to our allowed domain
        parsed_url = urllib.parse.urlparse(absolute_url)
        if parsed_url.netloc == ALLOWED_DOMAIN:
            links.append(absolute_url)
            
    return links


def clean_text(html: str) -> str:
    """
    Remove noise from the HTML and extract only readable text content.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # 1. Remove unwanted noisy tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "meta", "link", "iframe", "button", "form"]):
        tag.decompose()
        
    # Remove elements by class that are known noise (e.g., top menus, mobile menus, breadcrumbs)
    noisy_classes = [
        "menu", "dropdown", "navbar", "nav", "footer", "sidebar", "widget",
        "breadcrumb", "pagination", "social", "share", "cookie", "banner"
    ]
    for element in soup.find_all(class_=lambda c: c and any(noise in c.lower() for noise in noisy_classes)):
        element.decompose()
        
    # 2. Extract meaningful text from specific tags
    content_lines = []
    
    # Find all headings, paragraphs, and list items
    meaningful_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "span", "div"])
    
    for tag in meaningful_tags:
        # Get text, strip whitespace, and handle multiple spaces/newlines
        text = tag.get_text(separator=' ', strip=True)
        # Ignore very short snippets and Javascript injection artifacts
        if text and len(text) > 5 and not text.startswith("Loading..."):
            # Prevent adding the exact same line back-to-back
            if not content_lines or content_lines[-1] != text:
                content_lines.append(text)
            
    # Combine everything into a single readable string
    final_text = "\n".join(content_lines)
    
    # Further cleanup of excessive newlines
    final_text = "\n".join([line for line in final_text.splitlines() if line.strip()])
    
    return final_text


def save_page(url: str, html: str) -> bool:
    """
    Extract title and content, organize as JSON, and save to data/raw/.
    """
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled Page"
    
    content = clean_text(html)
    
    # Skip pages that have little to no meaningful text
    if len(content) < 50:
        logger.info(f"Skipping {url} - too little readable text extracted.")
        return False
        
    slug = get_slug_from_url(url)
    filename = os.path.join(SAVE_DIR, f"{slug}.json")
    
    # Handle filename collisions
    original_filename = filename
    counter = 1
    while os.path.exists(filename):
        filename = original_filename.replace(".json", f"_{counter}.json")
        counter += 1
        
    data = {
        "url": url,
        "title": title,
        "content": content
    }
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Extracted {len(content)} characters. Saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return False


def crawl(start_url: str):
    """
    Execute breadth-first crawl starting from the given URL.
    """
    setup_directories()
    
    # Queue stores tuples of (url, current_depth)
    queue = deque([(start_url, 0)])
    
    # Set of normalized URLs we have already visited/queued
    visited = set([urllib.parse.urldefrag(start_url)[0]])
    
    # Keep track of how many pages we've successfully saved
    pages_saved = 0
    
    logger.info(f"Starting crawl at {start_url}")
    logger.info(f"Limits: MAX_DEPTH={MAX_DEPTH}, MAX_PAGES={MAX_PAGES}")
    
    while queue and pages_saved < MAX_PAGES:
        current_url, current_depth = queue.popleft()
        
        logger.info(f"Scraping page: {current_url} (Depth: {current_depth})")
        
        html = fetch_page(current_url)
        if html:
            # Process and save the page
            saved = save_page(current_url, html)
            if saved:
                pages_saved += 1
                
            # If we haven't reached max depth, find new links to queue
            if current_depth < MAX_DEPTH:
                links = extract_links(html, current_url)
                
                new_links_queued = 0
                for link in links:
                    if link not in visited:
                        visited.add(link)
                        queue.append((link, current_depth + 1))
                        new_links_queued += 1
                        
                # logger.info(f"Found {new_links_queued} new links on {current_url}")
                
        # Politeness policy: delay between requests
        time.sleep(1)
        
    logger.info(f"Crawl finished. Successfully processed and saved {pages_saved} pages out of {len(visited)} visited urls.")


if __name__ == "__main__":
    crawl(START_URL)
