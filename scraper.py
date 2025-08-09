#!/usr/bin/env python3
"""
Mathematical Theorem Scraper
Scrapes theorem data from multiple sources and stores in SQLite database
"""

import sqlite3
import requests
from bs4 import BeautifulSoup
import arxiv
import fitz  # PyMuPDF
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import os
from datetime import datetime

# DDL for Database
def init_db(db_path='theorems.db'):
    """Initialize SQLite database with theorems table"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS theorems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            year INTEGER,
            description TEXT,
            formula TEXT,
            era TEXT,
            source TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

# Era Categorization
def get_era(year):
    """Categorize theorem by era based on year"""
    if year is None:
        return 'unknown'
    if year < 1900:
        return 'pre-1900'
    elif 1900 <= year <= 1999:
        return '1900s'
    else:
        return 'today'

# Helper function to extract year from text
def extract_year(text):
    """Extract the first 4-digit year from text"""
    year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9]|[1-9][0-9]{2})\b', text)
    return int(year_match.group()) if year_match else None

# Scraper Functions
def scrape_mactutor():
    """Scrape MacTutor History of Mathematics"""
    print("Scraping MacTutor...")
    url = 'https://mathshistory.st-andrews.ac.uk/HistTopics/'
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        theorems = []
        
        # Find all history topic links
        for link in soup.find_all('a', href=True):
            if '/HistTopics/' in link['href'] and link.text.strip():
                name = link.text.strip()
                # Try to get more details by following the link
                detail_url = 'https://mathshistory.st-andrews.ac.uk' + link['href']
                try:
                    detail_response = requests.get(detail_url, timeout=5)
                    detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                    # Extract year from content
                    year = extract_year(detail_soup.text)
                    # Get first paragraph as description
                    desc_elem = detail_soup.find('p')
                    desc = desc_elem.text.strip()[:500] if desc_elem else name
                except:
                    year = None
                    desc = name
                
                theorems.append({
                    'name': name,
                    'year': year,
                    'desc': desc,
                    'formula': None,
                    'category': 'history',
                    'source': 'MacTutor'
                })
                
                if len(theorems) >= 10:  # Limit for demo
                    break
        
        print(f"Found {len(theorems)} theorems from MacTutor")
        return theorems
    except Exception as e:
        print(f"Error scraping MacTutor: {e}")
        return []

def scrape_wikipedia():
    """Scrape Wikipedia Timeline of Mathematics"""
    print("Scraping Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/Timeline_of_mathematics'
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        theorems = []
        
        # Find timeline entries
        for li in soup.find_all('li'):
            text = li.text.strip()
            # Look for entries with years and mathematical content
            if re.search(r'^\d+', text) or re.search(r'\d{3,4}\s*(BC|AD|CE|BCE)?', text):
                year = extract_year(text)
                # Extract theorem name/description
                if '–' in text:
                    parts = text.split('–', 1)
                    if len(parts) > 1:
                        name = parts[1].split('.')[0].strip()[:200]
                    else:
                        name = text[:200]
                else:
                    name = text[:200]
                
                if 'theorem' in text.lower() or 'proof' in text.lower() or 'formula' in text.lower():
                    theorems.append({
                        'name': name,
                        'year': year,
                        'desc': text[:500],
                        'formula': None,
                        'category': 'general',
                        'source': 'Wikipedia'
                    })
                    
                    if len(theorems) >= 15:  # Limit
                        break
        
        print(f"Found {len(theorems)} theorems from Wikipedia")
        return theorems
    except Exception as e:
        print(f"Error scraping Wikipedia: {e}")
        return []

def scrape_mathscinet():
    """Placeholder for MathSciNet (requires authentication)"""
    print("MathSciNet scraping skipped (requires authentication)")
    # In production, you would add authentication here
    # Example structure for authenticated scraping:
    theorems = [
        {
            'name': 'Sample MathSciNet Theorem',
            'year': 2020,
            'desc': 'This would contain actual theorem data from MathSciNet',
            'formula': None,
            'category': 'research',
            'source': 'MathSciNet'
        }
    ]
    return []  # Return empty for now

def scrape_zbmath():
    """Scrape zbMATH Open"""
    print("Scraping zbMATH...")
    # zbMATH Open API endpoint
    url = 'https://zbmath.org/api/v1/document/_search'
    headers = {'Accept': 'application/json'}
    params = {
        'q': 'theorem',
        'size': 10
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        theorems = []
        
        if response.status_code == 200:
            data = response.json()
            for hit in data.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                name = source.get('title', 'Unknown')
                year = source.get('year')
                desc = source.get('abstract', '')[:500] if source.get('abstract') else name
                
                theorems.append({
                    'name': name,
                    'year': year,
                    'desc': desc,
                    'formula': None,
                    'category': 'mathematics',
                    'source': 'zbMATH'
                })
        
        print(f"Found {len(theorems)} theorems from zbMATH")
        return theorems
    except Exception as e:
        print(f"Error scraping zbMATH: {e}")
        return []

def scrape_arxiv():
    """Scrape arXiv for mathematical theorems"""
    print("Scraping arXiv...")
    try:
        client = arxiv.Client()
        # Search for papers about theorems in math history
        search = arxiv.Search(
            query='cat:math.HO OR (theorem AND mathematics)',
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        theorems = []
        for result in client.results(search):
            name = result.title
            year = result.published.year if result.published else None
            desc = result.summary[:500]
            
            theorems.append({
                'name': name,
                'year': year,
                'desc': desc,
                'formula': None,
                'category': 'research',
                'source': 'arXiv'
            })
        
        print(f"Found {len(theorems)} theorems from arXiv")
        return theorems
    except Exception as e:
        print(f"Error scraping arXiv: {e}")
        return []

def scrape_eudml():
    """Scrape European Digital Mathematics Library"""
    print("Scraping EuDML...")
    url = 'https://eudml.org/search'
    params = {'q': 'theorem', 'rows': 10}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        theorems = []
        
        # Find search results
        for result in soup.find_all('div', class_='result'):
            title_elem = result.find('a', class_='title')
            if title_elem:
                name = title_elem.text.strip()
                year = extract_year(result.text)
                desc_elem = result.find('div', class_='snippet')
                desc = desc_elem.text.strip()[:500] if desc_elem else name
                
                theorems.append({
                    'name': name,
                    'year': year,
                    'desc': desc,
                    'formula': None,
                    'category': 'mathematics',
                    'source': 'EuDML'
                })
        
        print(f"Found {len(theorems)} theorems from EuDML")
        return theorems
    except Exception as e:
        print(f"Error scraping EuDML: {e}")
        return []

def scrape_harvard_pdf():
    """Scrape Harvard mathematics PDF"""
    print("Scraping Harvard PDF...")
    url = 'https://people.math.harvard.edu/~knill/graphgeometry/papers/fundamental.pdf'
    
    try:
        response = requests.get(url, timeout=20)
        temp_pdf = 'temp_harvard.pdf'
        
        with open(temp_pdf, 'wb') as f:
            f.write(response.content)
        
        doc = fitz.open(temp_pdf)
        theorems = []
        full_text = ''
        
        for page in doc:
            full_text += page.get_text()
        
        # Look for theorem patterns
        theorem_pattern = re.compile(r'(Theorem\s+\d+\.?\d*|Fundamental\s+Theorem|.*\'s\s+Theorem)', re.IGNORECASE)
        lines = full_text.split('\n')
        
        for i, line in enumerate(lines):
            if theorem_pattern.search(line):
                name = line.strip()[:200]
                # Look for year in surrounding context
                context = ' '.join(lines[max(0, i-2):min(len(lines), i+3)])
                year = extract_year(context)
                desc = context[:500]
                
                theorems.append({
                    'name': name,
                    'year': year,
                    'desc': desc,
                    'formula': None,
                    'category': 'geometry',
                    'source': 'Harvard PDF'
                })
                
                if len(theorems) >= 10:
                    break
        
        doc.close()
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        
        print(f"Found {len(theorems)} theorems from Harvard PDF")
        return theorems
    except Exception as e:
        print(f"Error scraping Harvard PDF: {e}")
        return []

def scrape_mathigon():
    """Scrape Mathigon Timeline using Selenium"""
    print("Scraping Mathigon...")
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        url = 'https://mathigon.org/timeline'
        driver.get(url)
        
        # Wait for content to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'event')))
        time.sleep(2)  # Additional wait for JS rendering
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        
        theorems = []
        for event in soup.find_all('div', class_='event'):
            text = event.text.strip()
            year = extract_year(text)
            title_elem = event.find('h3')
            name = title_elem.text.strip() if title_elem else text.split('\n')[0][:200]
            desc = text[:500]
            
            if 'theorem' in text.lower() or 'proof' in text.lower():
                theorems.append({
                    'name': name,
                    'year': year,
                    'desc': desc,
                    'formula': None,
                    'category': 'timeline',
                    'source': 'Mathigon'
                })
                
                if len(theorems) >= 10:
                    break
        
        print(f"Found {len(theorems)} theorems from Mathigon")
        return theorems
    except Exception as e:
        print(f"Error scraping Mathigon: {e}")
        return []

def display_stats(conn):
    """Display statistics about scraped data"""
    c = conn.cursor()
    
    # Total theorems
    c.execute("SELECT COUNT(*) FROM theorems")
    total = c.fetchone()[0]
    print(f"\n{'='*50}")
    print(f"Total theorems scraped: {total}")
    
    # By era
    print("\nTheorems by era:")
    c.execute("SELECT era, COUNT(*) FROM theorems GROUP BY era")
    for era, count in c.fetchall():
        print(f"  {era}: {count}")
    
    # By source
    print("\nTheorems by source:")
    c.execute("SELECT source, COUNT(*) FROM theorems GROUP BY source")
    for source, count in c.fetchall():
        print(f"  {source}: {count}")
    
    # Sample theorems
    print("\nSample theorems:")
    c.execute("SELECT name, year, source FROM theorems LIMIT 5")
    for name, year, source in c.fetchall():
        print(f"  - {name[:50]}... ({year}) from {source}")
    
    print(f"{'='*50}\n")

# Main execution
def main():
    """Main scraping execution"""
    print("Mathematical Theorem Scraper")
    print("="*50)
    print(f"Starting at {datetime.now()}")
    
    # Initialize database
    conn = init_db()
    c = conn.cursor()
    
    # List of scraper functions
    scrapers = [
        scrape_mactutor,
        scrape_wikipedia,
        scrape_mathscinet,
        scrape_zbmath,
        scrape_arxiv,
        scrape_eudml,
        scrape_harvard_pdf,
        # scrape_mathigon  # Comment out if Selenium not set up
    ]
    
    total_theorems = 0
    
    # Run each scraper
    for scraper in scrapers:
        try:
            print(f"\nRunning {scraper.__name__}...")
            theorems = scraper()
            
            # Insert theorems into database
            for th in theorems:
                era = get_era(th['year'])
                c.execute('''
                    INSERT INTO theorems (name, year, description, formula, era, source, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    th['name'],
                    th['year'],
                    th['desc'],
                    th['formula'],
                    era,
                    th['source'],
                    th['category']
                ))
            
            conn.commit()
            total_theorems += len(theorems)
            
        except Exception as e:
            print(f"Error in {scraper.__name__}: {e}")
            continue
    
    # Display statistics
    display_stats(conn)
    
    # Close database
    conn.close()
    
    print(f"Scraping complete at {datetime.now()}")
    print(f"Total theorems added: {total_theorems}")
    print("Database saved as: theorems.db")

if __name__ == '__main__':
    main()