#!/usr/bin/env python3
"""
Google Patents API Integration

This module provides real patent search capabilities using Google Patents
to replace the mock data in the patent chatbot.
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import quote_plus
import re
from bs4 import BeautifulSoup
import random

# Selenium imports (optional)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PatentResult:
    """Container for patent search results"""
    patent_number: str
    title: str
    abstract: Optional[str] = None
    inventors: Optional[List[str]] = None
    assignee: Optional[str] = None
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None
    status: Optional[str] = None
    classification_codes: Optional[List[str]] = None
    claims_count: Optional[int] = None
    source: str = "Google Patents"
    url: Optional[str] = None
    main_ipc_code: Optional[str] = None

class GooglePatentsAPI:
    """
    Google Patents API integration using web scraping
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, use_selenium: bool = False):
        self.base_url = "https://patents.google.com"
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.session = requests.Session()
        
        # Set user agent to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Initialize Selenium if requested and available
        self.driver = None
        if self.use_selenium:
            try:
                chrome_options = Options()
                chrome_options.add_argument('--headless')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-plugins')
                chrome_options.add_argument('--disable-images')
                chrome_options.add_argument('--disable-javascript')  # Disable JS for faster loading
                chrome_options.add_argument('--disable-web-security')
                chrome_options.add_argument('--allow-running-insecure-content')
                chrome_options.add_argument('--disable-background-timer-throttling')
                chrome_options.add_argument('--disable-backgrounding-occluded-windows')
                chrome_options.add_argument('--disable-renderer-backgrounding')
                chrome_options.add_argument('--disable-features=TranslateUI')
                chrome_options.add_argument('--disable-ipc-flooding-protection')
                
                self.driver = webdriver.Chrome(options=chrome_options)
                self.driver.set_page_load_timeout(30)
                self.driver.implicitly_wait(10)
                logger.info("✅ Selenium WebDriver initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Selenium: {e}")
                self.use_selenium = False
        elif use_selenium and not SELENIUM_AVAILABLE:
            logger.warning("⚠️ Selenium not available, falling back to requests")
            self.use_selenium = False
    
    def search_patents(self, query: str, max_results: int = 10) -> List[PatentResult]:
        """
        Search for patents using Google Patents
        
        Args:
            query: Search query (patent number, title, or keywords)
            max_results: Maximum number of results to return
            
        Returns:
            List of PatentResult objects
        """
        try:
            logger.info(f"Searching Google Patents for: {query}")
            
            if self.use_selenium and self.driver:
                return self._search_with_selenium(query, max_results)
            else:
                return self._search_with_requests(query, max_results)
                
        except Exception as e:
            logger.error(f"Error searching patents for '{query}': {e}")
            return []
    
    def _search_with_selenium(self, query: str, max_results: int) -> List[PatentResult]:
        """Search using Selenium for JavaScript-rendered content"""
        try:
            # Build search URL
            search_url = self._build_search_url(query)
            
            # Navigate to page
            self.driver.get(search_url)
            
            # Wait for results to load with better error handling
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
                )
            except Exception as e:
                logger.warning(f"Timeout waiting for results, continuing anyway: {e}")
            
            # Give extra time for all results to load
            time.sleep(3)
            
            # Get page source after JavaScript execution
            page_source = self.driver.page_source
            
            # Parse results
            patents = self._parse_search_results(page_source, max_results)
            
            logger.info(f"Selenium search found {len(patents)} patents")
            return patents
            
        except Exception as e:
            logger.error(f"Selenium search failed: {e}")
            return []
        finally:
            # Always cleanup after search
            try:
                if self.driver:
                    self.driver.delete_all_cookies()
            except Exception as e:
                logger.warning(f"Error cleaning up Selenium session: {e}")
    
    def _search_with_requests(self, query: str, max_results: int) -> List[PatentResult]:
        """Search using requests (fallback method)"""
        try:
            # Rate limiting
            self._rate_limit()
            
            # Build search URL
            search_url = self._build_search_url(query)
            
            # Make request
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse results
            patents = self._parse_search_results(response.text, max_results)
            
            logger.info(f"Requests search found {len(patents)} patents")
            return patents
            
        except Exception as e:
            logger.error(f"Requests search failed: {e}")
            return []
    
    def get_patent_details(self, patent_number: str) -> Optional[PatentResult]:
        """
        Get detailed information for a specific patent
        
        Args:
            patent_number: Patent number (e.g., "US10438354B2")
            
        Returns:
            PatentResult with detailed information or None if not found
        """
        try:
            logger.info(f"Getting details for patent: {patent_number}")
            
            # Rate limiting
            self._rate_limit()
            
            # Validate patent number format
            if not self._is_valid_patent_number(patent_number):
                logger.warning(f"Invalid patent number format: {patent_number}")
                return None
            
            # Build patent URL
            patent_url = f"{self.base_url}/patent/{patent_number}"
            
            # Make request
            response = self.session.get(patent_url, timeout=self.timeout)
            
            # Handle 404 errors gracefully
            if response.status_code == 404:
                logger.warning(f"Patent not found in Google Patents: {patent_number}")
                return None
            
            response.raise_for_status()
            
            # Parse patent details
            patent = self._parse_patent_details(response.text, patent_number)
            
            if patent:
                logger.info(f"Successfully retrieved details for patent: {patent_number}")
            else:
                logger.warning(f"No details found for patent: {patent_number}")
            
            return patent
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Patent not found in Google Patents: {patent_number}")
                return None
            else:
                logger.error(f"HTTP error getting patent details: {e}")
                return None
        except Exception as e:
            logger.error(f"Error getting patent details: {e}")
            return None
    
    def _build_search_url(self, query: str) -> str:
        """Build Google Patents search URL"""
        encoded_query = quote_plus(query)
        return f"{self.base_url}/?q={encoded_query}&language=ENGLISH&type=PATENT"
    
    def _rate_limit(self):
        """Implement rate limiting to avoid being blocked"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _parse_search_results(self, html_content: str, max_results: int) -> List[PatentResult]:
        """Parse Google Patents search results from HTML"""
        patents = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Method 1: Look for patent result containers
            patent_elements = soup.find_all('article', class_='result')
            
            # Method 2: Look for any article elements
            if not patent_elements:
                patent_elements = soup.find_all('article')
            
            # Method 3: Look for patent links
            if not patent_elements:
                patent_links = soup.find_all('a', href=re.compile(r'/patent/'))
                for link in patent_links[:max_results]:
                    patent_number = self._extract_patent_number_from_url(link.get('href', ''))
                    if patent_number:
                        patent = PatentResult(
                            patent_number=patent_number,
                            title=link.get_text(strip=True) or f"Patent {patent_number}",
                            url=f"{self.base_url}{link.get('href')}"
                        )
                        patents.append(patent)
            
            # Method 4: Extract from structured elements
            for element in patent_elements[:max_results]:
                patent = self._extract_patent_from_element(element)
                if patent:
                    patents.append(patent)
            
            # Method 5: Alternative parsing for JavaScript-rendered content
            if not patents:
                patents = self._parse_alternative_results(html_content, max_results)
            
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
        
        return patents
    
    def _extract_patent_from_element(self, element) -> Optional[PatentResult]:
        """Extract patent information from a search result element"""
        try:
            # Extract patent number
            patent_link = element.find('a', href=True)
            if not patent_link:
                return None
            
            href = patent_link['href']
            patent_number = self._extract_patent_number_from_url(href)
            if not patent_number:
                return None
            
            # Extract title - try multiple selectors
            title = "Unknown Title"
            title_selectors = [
                'span[itemprop="title"]',
                'h3',
                'h2', 
                'h1',
                '.title',
                '.patent-title'
            ]
            
            for selector in title_selectors:
                title_element = element.select_one(selector)
                if title_element:
                    title_text = title_element.get_text(strip=True)
                    if title_text and title_text != patent_number:
                        title = title_text
                        break
            
            # If no title found, use a generic one
            if title == "Unknown Title":
                title = f"Patent {patent_number}"
            
            # Extract abstract - try multiple selectors
            abstract = None
            abstract_selectors = [
                'span[itemprop="abstract"]',
                '.abstract',
                '.patent-abstract',
                'p',
                '.description'
            ]
            
            for selector in abstract_selectors:
                abstract_element = element.select_one(selector)
                if abstract_element:
                    abstract_text = abstract_element.get_text(strip=True)
                    if abstract_text and len(abstract_text) > 20 and abstract_text != patent_number:
                        abstract = abstract_text
                        break
            
            # If no abstract found, create a generic one
            if not abstract:
                abstract = f"Patent {patent_number} - {title}"
            
            # Extract inventors
            inventors = []
            inventor_elements = element.find_all('span', itemprop='inventor')
            for inv in inventor_elements:
                inventors.append(inv.get_text(strip=True))
            
            # Extract assignee
            assignee_element = element.find('span', itemprop='assignee')
            assignee = assignee_element.get_text(strip=True) if assignee_element else None
            
            # Extract dates
            filing_date = None
            publication_date = None
            date_elements = element.find_all('time')
            for date_elem in date_elements:
                date_text = date_elem.get_text(strip=True)
                if 'filing' in date_text.lower():
                    filing_date = date_text
                elif 'publication' in date_text.lower():
                    publication_date = date_text
            
            # Build URL
            url = f"{self.base_url}{href}" if href.startswith('/') else href
            
            return PatentResult(
                patent_number=patent_number,
                title=title,
                abstract=abstract,
                inventors=inventors if inventors else None,
                assignee=assignee,
                filing_date=filing_date,
                publication_date=publication_date,
                url=url
            )
            
        except Exception as e:
            logger.error(f"Error extracting patent from element: {e}")
            return None
    
    def _parse_alternative_results(self, html_content: str, max_results: int) -> List[PatentResult]:
        """Alternative parsing method for when structured elements aren't found"""
        patents = []
        
        try:
            # Look for patent numbers in the HTML
            patent_pattern = r'patent/([A-Z]{2}\d+[A-Z]?\d*)'
            patent_numbers = re.findall(patent_pattern, html_content)
            
            # Look for titles
            title_pattern = r'<span[^>]*itemprop="title"[^>]*>([^<]+)</span>'
            titles = re.findall(title_pattern, html_content)
            
            # Look for abstracts
            abstract_pattern = r'<span[^>]*itemprop="abstract"[^>]*>([^<]+)</span>'
            abstracts = re.findall(abstract_pattern, html_content)
            
            # Combine results
            for i, patent_number in enumerate(patent_numbers[:max_results]):
                title = titles[i] if i < len(titles) else f"Patent {patent_number}"
                abstract = abstracts[i] if i < len(abstracts) else None
                
                patent = PatentResult(
                    patent_number=patent_number,
                    title=title,
                    abstract=abstract,
                    url=f"{self.base_url}/patent/{patent_number}"
                )
                patents.append(patent)
                
        except Exception as e:
            logger.error(f"Error in alternative parsing: {e}")
        
        return patents
    
    def _parse_patent_details(self, html_content: str, patent_number: str) -> Optional[PatentResult]:
        """Parse detailed patent information from patent page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title_element = soup.find('span', itemprop='title')
            title = title_element.get_text(strip=True) if title_element else f"Patent {patent_number}"
            
            # Extract abstract
            abstract_element = soup.find('span', itemprop='abstract')
            abstract = abstract_element.get_text(strip=True) if abstract_element else None
            
            # Extract inventors
            inventors = []
            inventor_elements = soup.find_all('span', itemprop='inventor')
            for inv in inventor_elements:
                inventors.append(inv.get_text(strip=True))
            
            # Extract assignee
            assignee_element = soup.find('span', itemprop='assignee')
            assignee = assignee_element.get_text(strip=True) if assignee_element else None
            
            # Extract dates
            filing_date = None
            publication_date = None
            date_elements = soup.find_all('time')
            for date_elem in date_elements:
                date_text = date_elem.get_text(strip=True)
                if 'filing' in date_text.lower():
                    filing_date = date_text
                elif 'publication' in date_text.lower():
                    publication_date = date_text
            
            # Extract classification codes - focus on main IPC code
            classification_codes = []
            main_ipc_code = None
            
            # Look for IPC classification codes
            class_elements = soup.find_all('span', class_='classification')
            for class_elem in class_elements:
                code_text = class_elem.get_text(strip=True)
                classification_codes.append(code_text)
                
                # Extract main IPC code (e.g., G06N, G06V)
                if code_text.startswith('G06'):
                    # Look for the main IPC code pattern
                    ipc_match = re.search(r'G06[NV]\d+/\d+', code_text)
                    if ipc_match:
                        main_ipc_code = ipc_match.group(0)
                        break
            
            # If no main IPC found in classification elements, search in the entire content
            if not main_ipc_code:
                ipc_pattern = r'G06[NV]\d+/\d+'
                ipc_matches = re.findall(ipc_pattern, html_content)
                if ipc_matches:
                    main_ipc_code = ipc_matches[0]
            
            # Extract claims count
            claims_count = None
            claims_element = soup.find('span', string=re.compile(r'claims?', re.I))
            if claims_element:
                claims_text = claims_element.get_text()
                claims_match = re.search(r'(\d+)', claims_text)
                if claims_match:
                    claims_count = int(claims_match.group(1))
            
            # Determine status based on content
            status = self._determine_patent_status(html_content)
            
            return PatentResult(
                patent_number=patent_number,
                title=title,
                abstract=abstract,
                inventors=inventors if inventors else None,
                assignee=assignee,
                filing_date=filing_date,
                publication_date=publication_date,
                status=status,
                classification_codes=classification_codes if classification_codes else None,
                claims_count=claims_count,
                url=f"{self.base_url}/patent/{patent_number}",
                main_ipc_code=main_ipc_code
            )
            
        except Exception as e:
            logger.error(f"Error parsing patent details: {e}")
            return None
    
    def _extract_patent_number_from_url(self, url: str) -> Optional[str]:
        """Extract patent number from Google Patents URL"""
        try:
            # Pattern: /patent/US10438354B2/en
            match = re.search(r'/patent/([A-Z]{2}\d+[A-Z]?\d*)', url)
            if match:
                return match.group(1)
            
            # Alternative pattern: /patent/US10438354B2
            match = re.search(r'/patent/([A-Z]{2}\d+[A-Z]?\d*)$', url)
            if match:
                return match.group(1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting patent number from URL: {e}")
            return None
    
    def _determine_patent_status(self, html_content: str) -> str:
        """Determine patent status from page content"""
        content_lower = html_content.lower()
        
        if 'granted' in content_lower or 'issued' in content_lower:
            return "GRANTED"
        elif 'pending' in content_lower or 'application' in content_lower:
            return "PENDING"
        elif 'expired' in content_lower:
            return "EXPIRED"
        elif 'abandoned' in content_lower:
            return "ABANDONED"
        else:
            return "UNKNOWN"
    
    def search_by_patent_number(self, patent_number: str) -> Optional[PatentResult]:
        """
        Search for a specific patent by patent number
        
        Args:
            patent_number: Patent number (e.g., "US10438354B2")
            
        Returns:
            PatentResult if found, None otherwise
        """
        return self.get_patent_details(patent_number)
    
    def search_by_keywords(self, keywords: str, max_results: int = 10) -> List[PatentResult]:
        """
        Search for patents by keywords
        
        Args:
            keywords: Search keywords
            max_results: Maximum number of results
            
        Returns:
            List of PatentResult objects
        """
        return self.search_patents(keywords, max_results)
    
    def search_by_technology_area(self, technology: str, max_results: int = 10) -> List[PatentResult]:
        """
        Search for patents in a specific technology area
        
        Args:
            technology: Technology area (e.g., "machine learning", "blockchain")
            max_results: Maximum number of results
            
        Returns:
            List of PatentResult objects
        """
        return self.search_patents(technology, max_results)
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                # Close all windows
                self.driver.close()
                # Quit the driver
                self.driver.quit()
                logger.info("✅ Selenium WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")
            finally:
                self.driver = None

    def _is_valid_patent_number(self, patent_number: str) -> bool:
        """Validate patent number format"""
        import re
        
        # Common patent number patterns
        patterns = [
            r'^[A-Z]{2}\d+[A-Z]?\d*$',  # US10438354B2, EP1234567A1
            r'^[A-Z]{1,2}\d+[A-Z]?\d*$',  # US12345678, CN123456789
            r'^[A-Z]{2,3}\d+[A-Z]?\d*$',  # WO123456789, JP123456789
        ]
        
        for pattern in patterns:
            if re.match(pattern, patent_number):
                return True
        
        return False

# Convenience functions for easy integration
def search_google_patents(query: str, max_results: int = 10, use_selenium: bool = False) -> List[Dict]:
    """
    Convenience function to search Google Patents and return results as dictionaries
    
    Args:
        query: Search query
        max_results: Maximum number of results
        use_selenium: Whether to use Selenium for JavaScript-rendered content
        
    Returns:
        List of patent dictionaries compatible with existing chatbot interface
    """
    api = GooglePatentsAPI(use_selenium=use_selenium)
    
    try:
        patents = api.search_patents(query, max_results)
        
        # Convert to dictionary format compatible with existing interface
        results = []
        for patent in patents:
            results.append({
                "title": patent.title,
                "patent_number": patent.patent_number,
                "abstract": patent.abstract or f"Patent {patent.patent_number} - {patent.title}",
                "status": patent.status or "UNKNOWN",
                "source": patent.source,
                "url": patent.url,
                "inventors": patent.inventors,
                "assignee": patent.assignee,
                "filing_date": patent.filing_date,
                "publication_date": patent.publication_date,
                "classification_codes": patent.classification_codes,
                "claims_count": patent.claims_count,
                "main_ipc_code": patent.main_ipc_code
            })
        
        return results
    finally:
        api.cleanup()

def get_patent_details(patent_number: str) -> Optional[Dict]:
    """
    Convenience function to get detailed patent information
    
    Args:
        patent_number: Patent number
        
    Returns:
        Patent dictionary or None if not found
    """
    api = GooglePatentsAPI()
    
    try:
        patent = api.get_patent_details(patent_number)
        
        if patent:
            return {
                "title": patent.title,
                "patent_number": patent.patent_number,
                "abstract": patent.abstract or f"Patent {patent.patent_number} - {patent.title}",
                "status": patent.status or "UNKNOWN",
                "source": patent.source,
                "url": patent.url,
                "inventors": patent.inventors,
                "assignee": patent.assignee,
                "filing_date": patent.filing_date,
                "publication_date": patent.publication_date,
                "classification_codes": patent.classification_codes,
                "claims_count": patent.claims_count,
                "main_ipc_code": patent.main_ipc_code
            }
        
        return None
    finally:
        api.cleanup() 