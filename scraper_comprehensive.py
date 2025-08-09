#!/usr/bin/env python3
"""
Comprehensive Mathematical Theorem Scraper
Designed to collect thousands to hundreds of thousands of theorems
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
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from urllib.parse import urljoin, urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DDL for Enhanced Database
def init_db(db_path='theorems_comprehensive.db'):
    """Initialize SQLite database with enhanced schema"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS theorems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            year INTEGER,
            description TEXT,
            formula TEXT,
            proof_available BOOLEAN DEFAULT 0,
            authors TEXT,
            field TEXT,
            subfield TEXT,
            era TEXT,
            source TEXT,
            source_url TEXT,
            category TEXT,
            hash TEXT UNIQUE,  -- To prevent duplicates
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_year ON theorems(year)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_era ON theorems(era)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_field ON theorems(field)')
    conn.commit()
    return conn

# Era Categorization
def get_era(year):
    """Categorize theorem by era based on year"""
    if year is None:
        return 'unknown'
    if year < 0:
        return 'ancient'
    elif year < 500:
        return 'classical'
    elif year < 1500:
        return 'medieval'
    elif year < 1900:
        return 'early-modern'
    elif year < 2000:
        return '20th-century'
    else:
        return '21st-century'

# Helper functions
def extract_year(text):
    """Extract year from text with BC/AD handling"""
    # Check for BC dates
    bc_match = re.search(r'(\d+)\s*BC', text, re.IGNORECASE)
    if bc_match:
        return -int(bc_match.group(1))
    
    # Standard year extraction
    year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9]|[1-9][0-9]{2})\b', text)
    return int(year_match.group()) if year_match else None

def generate_hash(name, source):
    """Generate unique hash for theorem to prevent duplicates"""
    content = f"{name}_{source}".lower().strip()
    return hashlib.md5(content.encode()).hexdigest()

def insert_theorem(conn, theorem_data):
    """Insert theorem with duplicate checking"""
    c = conn.cursor()
    theorem_hash = generate_hash(theorem_data['name'], theorem_data['source'])
    
    try:
        c.execute('''
            INSERT OR IGNORE INTO theorems 
            (name, year, description, formula, proof_available, authors, 
             field, subfield, era, source, source_url, category, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            theorem_data.get('name'),
            theorem_data.get('year'),
            theorem_data.get('desc'),
            theorem_data.get('formula'),
            theorem_data.get('proof_available', 0),
            theorem_data.get('authors'),
            theorem_data.get('field'),
            theorem_data.get('subfield'),
            get_era(theorem_data.get('year')),
            theorem_data.get('source'),
            theorem_data.get('source_url'),
            theorem_data.get('category'),
            theorem_hash
        ))
        conn.commit()
        return c.rowcount > 0
    except Exception as e:
        logger.error(f"Error inserting theorem: {e}")
        return False

# COMPREHENSIVE SCRAPERS

def scrape_proofwiki(max_pages=100):
    """Scrape ProofWiki - contains thousands of theorems with proofs"""
    logger.info("Scraping ProofWiki...")
    theorems = []
    base_url = 'https://proofwiki.org'
    
    # Categories to scrape
    categories = [
        '/wiki/Category:Theorems',
        '/wiki/Category:Lemmas',
        '/wiki/Category:Named_Theorems',
        '/wiki/Category:Fundamental_Theorems',
        '/wiki/Category:Famous_Theorems'
    ]
    
    for category_path in categories:
        try:
            url = base_url + category_path
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all theorem links
            content_div = soup.find('div', {'id': 'mw-pages'})
            if content_div:
                for link in content_div.find_all('a'):
                    if link.get('href', '').startswith('/wiki/') and ':' not in link.get('href', ''):
                        theorem_url = base_url + link['href']
                        name = link.text.strip()
                        
                        # Get theorem details
                        try:
                            detail_response = requests.get(theorem_url, timeout=5)
                            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                            
                            # Extract content
                            content = detail_soup.find('div', {'id': 'mw-content-text'})
                            if content:
                                desc = ' '.join(content.text.split()[:100])  # First 100 words
                                year = extract_year(content.text)
                                
                                # Check for formulas (LaTeX)
                                formula = None
                                math_elements = content.find_all(['math', 'span'], class_='tex')
                                if math_elements:
                                    formula = math_elements[0].text[:200]
                                
                                theorems.append({
                                    'name': name,
                                    'year': year,
                                    'desc': desc[:500],
                                    'formula': formula,
                                    'proof_available': 1,
                                    'source': 'ProofWiki',
                                    'source_url': theorem_url,
                                    'category': category_path.split(':')[-1]
                                })
                                
                                if len(theorems) >= max_pages:
                                    break
                        except:
                            pass
                        
                if len(theorems) >= max_pages:
                    break
                    
        except Exception as e:
            logger.error(f"Error in ProofWiki category {category_path}: {e}")
    
    logger.info(f"Found {len(theorems)} theorems from ProofWiki")
    return theorems

def scrape_mathworld(max_theorems=1000):
    """Scrape Wolfram MathWorld"""
    logger.info("Scraping MathWorld...")
    theorems = []
    base_url = 'https://mathworld.wolfram.com'
    
    # Alphabet browsing
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        try:
            url = f'{base_url}/letters/{letter}.html'
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a'):
                text = link.text.lower()
                if any(keyword in text for keyword in ['theorem', 'lemma', 'principle', 'law', 'rule', 'formula']):
                    name = link.text.strip()
                    href = link.get('href', '')
                    
                    if href and not href.startswith('http'):
                        detail_url = base_url + '/' + href
                        
                        try:
                            detail_response = requests.get(detail_url, timeout=5)
                            detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                            
                            # Extract description
                            content = detail_soup.find('div', class_='content')
                            if content:
                                paragraphs = content.find_all('p')
                                desc = ' '.join([p.text for p in paragraphs[:2]])[:500]
                            else:
                                desc = name
                            
                            year = extract_year(detail_soup.text)
                            
                            theorems.append({
                                'name': name,
                                'year': year,
                                'desc': desc,
                                'source': 'MathWorld',
                                'source_url': detail_url,
                                'category': 'mathematics'
                            })
                            
                            if len(theorems) >= max_theorems:
                                return theorems
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Error scraping MathWorld letter {letter}: {e}")
    
    logger.info(f"Found {len(theorems)} theorems from MathWorld")
    return theorems

def scrape_wikipedia_comprehensive():
    """Comprehensive Wikipedia scraping - multiple math pages"""
    logger.info("Scraping Wikipedia comprehensively...")
    theorems = []
    
    wikipedia_pages = [
        'https://en.wikipedia.org/wiki/List_of_theorems',
        'https://en.wikipedia.org/wiki/List_of_lemmas',
        'https://en.wikipedia.org/wiki/Timeline_of_mathematics',
        'https://en.wikipedia.org/wiki/List_of_mathematical_proofs',
        'https://en.wikipedia.org/wiki/Category:Theorems_in_algebra',
        'https://en.wikipedia.org/wiki/Category:Theorems_in_geometry',
        'https://en.wikipedia.org/wiki/Category:Theorems_in_analysis',
        'https://en.wikipedia.org/wiki/Category:Theorems_in_number_theory',
        'https://en.wikipedia.org/wiki/Category:Theorems_in_topology',
        'https://en.wikipedia.org/wiki/Category:Theorems_in_discrete_mathematics',
        'https://en.wikipedia.org/wiki/List_of_conjectures',
        'https://en.wikipedia.org/wiki/List_of_inequalities',
        'https://en.wikipedia.org/wiki/List_of_mathematical_identities'
    ]
    
    for page_url in wikipedia_pages:
        try:
            response = requests.get(page_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find content area
            content = soup.find('div', {'id': 'mw-content-text'})
            if not content:
                continue
            
            # Extract all list items that might be theorems
            for li in content.find_all('li'):
                text = li.text.strip()
                
                # Look for theorem patterns
                if any(keyword in text.lower() for keyword in 
                       ['theorem', 'lemma', 'law', 'principle', 'rule', 'formula', 'inequality', 'identity']):
                    
                    # Extract theorem name
                    link = li.find('a')
                    if link:
                        name = link.text.strip()
                        href = link.get('href', '')
                        source_url = 'https://en.wikipedia.org' + href if href.startswith('/wiki/') else page_url
                    else:
                        name = text[:200]
                        source_url = page_url
                    
                    year = extract_year(text)
                    
                    # Determine field
                    field = 'mathematics'
                    if 'algebra' in page_url.lower():
                        field = 'algebra'
                    elif 'geometry' in page_url.lower():
                        field = 'geometry'
                    elif 'analysis' in page_url.lower():
                        field = 'analysis'
                    elif 'number' in page_url.lower():
                        field = 'number theory'
                    elif 'topology' in page_url.lower():
                        field = 'topology'
                    
                    theorems.append({
                        'name': name,
                        'year': year,
                        'desc': text[:500],
                        'field': field,
                        'source': 'Wikipedia',
                        'source_url': source_url,
                        'category': 'encyclopedia'
                    })
            
            # Also check for tables of theorems
            for table in content.find_all('table'):
                for row in table.find_all('tr')[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        name = cells[0].text.strip()
                        if name and len(name) > 2:
                            desc = ' '.join([cell.text.strip() for cell in cells])[:500]
                            year = extract_year(desc)
                            
                            theorems.append({
                                'name': name,
                                'year': year,
                                'desc': desc,
                                'source': 'Wikipedia',
                                'source_url': page_url,
                                'category': 'encyclopedia'
                            })
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia page {page_url}: {e}")
    
    logger.info(f"Found {len(theorems)} theorems from Wikipedia")
    return theorems

def scrape_arxiv_comprehensive(max_results=5000):
    """Comprehensive arXiv scraping across all math categories"""
    logger.info("Scraping arXiv comprehensively...")
    theorems = []
    
    # All math categories in arXiv
    math_categories = [
        'math.AC',  # Commutative Algebra
        'math.AG',  # Algebraic Geometry
        'math.AP',  # Analysis of PDEs
        'math.AT',  # Algebraic Topology
        'math.CA',  # Classical Analysis and ODEs
        'math.CO',  # Combinatorics
        'math.CT',  # Category Theory
        'math.CV',  # Complex Variables
        'math.DG',  # Differential Geometry
        'math.DS',  # Dynamical Systems
        'math.FA',  # Functional Analysis
        'math.GM',  # General Mathematics
        'math.GN',  # General Topology
        'math.GR',  # Group Theory
        'math.GT',  # Geometric Topology
        'math.HO',  # History and Overview
        'math.IT',  # Information Theory
        'math.KT',  # K-Theory and Homology
        'math.LO',  # Logic
        'math.MG',  # Metric Geometry
        'math.MP',  # Mathematical Physics
        'math.NA',  # Numerical Analysis
        'math.NT',  # Number Theory
        'math.OA',  # Operator Algebras
        'math.OC',  # Optimization and Control
        'math.PR',  # Probability
        'math.QA',  # Quantum Algebra
        'math.RA',  # Rings and Algebras
        'math.RT',  # Representation Theory
        'math.SG',  # Symplectic Geometry
        'math.SP',  # Spectral Theory
        'math.ST'   # Statistics Theory
    ]
    
    client = arxiv.Client()
    results_per_category = max_results // len(math_categories)
    
    for category in math_categories:
        try:
            # Search for papers with theorem/lemma in title or abstract
            search = arxiv.Search(
                query=f'cat:{category} AND (ti:theorem OR ti:lemma OR abs:theorem OR abs:fundamental)',
                max_results=results_per_category,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in client.results(search):
                name = result.title
                year = result.published.year if result.published else None
                authors = ', '.join([author.name for author in result.authors])[:200]
                
                theorems.append({
                    'name': name,
                    'year': year,
                    'desc': result.summary[:500],
                    'authors': authors,
                    'field': category.replace('math.', ''),
                    'source': 'arXiv',
                    'source_url': result.entry_id,
                    'category': 'research'
                })
                
        except Exception as e:
            logger.error(f"Error scraping arXiv category {category}: {e}")
    
    logger.info(f"Found {len(theorems)} theorems from arXiv")
    return theorems

def scrape_oeis():
    """Scrape OEIS for sequence-related theorems"""
    logger.info("Scraping OEIS...")
    theorems = []
    
    # Search for sequences with theorem mentions
    base_url = 'https://oeis.org/search'
    keywords = ['theorem', 'fundamental', 'prime', 'fibonacci', 'euler', 'fermat']
    
    for keyword in keywords:
        try:
            params = {'q': keyword, 'fmt': 'json'}
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                for result in results[:100]:  # Limit per keyword
                    if 'name' in result:
                        name = result['name']
                        if 'theorem' in name.lower() or 'formula' in name.lower():
                            theorems.append({
                                'name': f"OEIS {result.get('number', '')}: {name}",
                                'desc': result.get('comment', '')[:500],
                                'formula': result.get('formula', ''),
                                'field': 'sequences',
                                'source': 'OEIS',
                                'source_url': f"https://oeis.org/{result.get('number', '')}",
                                'category': 'sequences'
                            })
        except Exception as e:
            logger.error(f"Error scraping OEIS for {keyword}: {e}")
    
    logger.info(f"Found {len(theorems)} theorems from OEIS")
    return theorems

def scrape_mactutor_comprehensive():
    """Deep scrape MacTutor with all biographies and topics"""
    logger.info("Scraping MacTutor comprehensively...")
    theorems = []
    base_url = 'https://mathshistory.st-andrews.ac.uk'
    
    # Scrape history topics
    topics_url = base_url + '/HistTopics/alphabetical.html'
    try:
        response = requests.get(topics_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a'):
            if link.get('href', '').startswith('../HistTopics/'):
                topic_url = base_url + link['href'].replace('..', '')
                name = link.text.strip()
                
                try:
                    topic_response = requests.get(topic_url, timeout=5)
                    topic_soup = BeautifulSoup(topic_response.text, 'html.parser')
                    
                    # Extract theorems mentioned
                    text = topic_soup.text
                    theorem_mentions = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:theorem|lemma|principle|law))', text)
                    
                    for theorem_name in theorem_mentions:
                        year = extract_year(text[max(0, text.index(theorem_name)-100):text.index(theorem_name)+100])
                        
                        theorems.append({
                            'name': theorem_name,
                            'year': year,
                            'desc': f"From MacTutor topic: {name}",
                            'source': 'MacTutor',
                            'source_url': topic_url,
                            'category': 'history'
                        })
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error scraping MacTutor topics: {e}")
    
    # Scrape biographies for theorems
    biographies_url = base_url + '/BiogIndex.html'
    try:
        response = requests.get(biographies_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a')[:500]:  # Limit to prevent timeout
            if link.get('href', '').startswith('Biographies/'):
                bio_url = base_url + '/' + link['href']
                mathematician = link.text.strip()
                
                try:
                    bio_response = requests.get(bio_url, timeout=5)
                    bio_soup = BeautifulSoup(bio_response.text, 'html.parser')
                    
                    # Look for theorem mentions
                    text = bio_soup.text
                    if 'theorem' in text.lower() or 'lemma' in text.lower():
                        # Try to extract specific theorem names
                        patterns = [
                            mathematician + "'s theorem",
                            mathematician + " theorem",
                            "theorem of " + mathematician
                        ]
                        
                        for pattern in patterns:
                            if pattern.lower() in text.lower():
                                year = extract_year(text)
                                
                                theorems.append({
                                    'name': pattern,
                                    'year': year,
                                    'desc': f"Associated with {mathematician}",
                                    'authors': mathematician,
                                    'source': 'MacTutor',
                                    'source_url': bio_url,
                                    'category': 'biography'
                                })
                                break
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error scraping MacTutor biographies: {e}")
    
    logger.info(f"Found {len(theorems)} theorems from MacTutor")
    return theorems

def scrape_encyclopedia_of_math():
    """Scrape Encyclopedia of Mathematics"""
    logger.info("Scraping Encyclopedia of Mathematics...")
    theorems = []
    base_url = 'https://encyclopediaofmath.org'
    
    # Categories to check
    categories = [
        '/wiki/Category:Mathematics',
        '/wiki/Category:Algebra',
        '/wiki/Category:Analysis',
        '/wiki/Category:Geometry',
        '/wiki/Category:Topology',
        '/wiki/Category:Number_theory'
    ]
    
    for category in categories:
        try:
            url = base_url + category
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all article links
            for link in soup.find_all('a'):
                if link.get('href', '').startswith('/wiki/') and ':' not in link.get('href', ''):
                    name = link.text.strip()
                    if any(kw in name.lower() for kw in ['theorem', 'lemma', 'principle', 'law']):
                        article_url = base_url + link['href']
                        
                        try:
                            article_response = requests.get(article_url, timeout=5)
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            
                            content = article_soup.find('div', {'id': 'content'})
                            if content:
                                desc = ' '.join(content.text.split()[:100])[:500]
                                year = extract_year(content.text)
                                
                                theorems.append({
                                    'name': name,
                                    'year': year,
                                    'desc': desc,
                                    'source': 'Encyclopedia of Mathematics',
                                    'source_url': article_url,
                                    'category': category.split(':')[-1]
                                })
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Error scraping Encyclopedia of Mathematics category {category}: {e}")
    
    logger.info(f"Found {len(theorems)} theorems from Encyclopedia of Mathematics")
    return theorems

def parallel_scraping(conn):
    """Run scrapers in parallel for speed"""
    logger.info("Starting parallel scraping...")
    
    scrapers = [
        (scrape_wikipedia_comprehensive, {}),
        (scrape_proofwiki, {'max_pages': 2000}),
        (scrape_mathworld, {'max_theorems': 2000}),
        (scrape_arxiv_comprehensive, {'max_results': 10000}),
        (scrape_oeis, {}),
        (scrape_mactutor_comprehensive, {}),
        (scrape_encyclopedia_of_math, {})
    ]
    
    total_inserted = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for scraper_func, kwargs in scrapers:
            future = executor.submit(scraper_func, **kwargs)
            futures.append((future, scraper_func.__name__))
        
        for future, scraper_name in futures:
            try:
                theorems = future.result(timeout=300)  # 5 minute timeout per scraper
                
                # Insert theorems into database
                for theorem in theorems:
                    if insert_theorem(conn, theorem):
                        total_inserted += 1
                
                logger.info(f"{scraper_name} completed: {len(theorems)} theorems found, {total_inserted} new inserted")
                
            except Exception as e:
                logger.error(f"Error in {scraper_name}: {e}")
    
    return total_inserted

def display_comprehensive_stats(conn):
    """Display comprehensive statistics"""
    c = conn.cursor()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE THEOREM DATABASE STATISTICS")
    print("="*60)
    
    # Total count
    c.execute("SELECT COUNT(*) FROM theorems")
    total = c.fetchone()[0]
    print(f"\nTOTAL THEOREMS: {total:,}")
    
    # By era
    print("\nTheorems by Era:")
    c.execute("SELECT era, COUNT(*) as cnt FROM theorems GROUP BY era ORDER BY cnt DESC")
    for era, count in c.fetchall():
        print(f"  {era:20} {count:8,} ({count*100/total:.1f}%)")
    
    # By source
    print("\nTheorems by Source:")
    c.execute("SELECT source, COUNT(*) as cnt FROM theorems GROUP BY source ORDER BY cnt DESC")
    for source, count in c.fetchall():
        print(f"  {source:25} {count:8,}")
    
    # By field
    print("\nTheorems by Field (top 10):")
    c.execute("SELECT field, COUNT(*) as cnt FROM theorems WHERE field IS NOT NULL GROUP BY field ORDER BY cnt DESC LIMIT 10")
    for field, count in c.fetchall():
        print(f"  {field:25} {count:8,}")
    
    # With proofs available
    c.execute("SELECT COUNT(*) FROM theorems WHERE proof_available = 1")
    with_proofs = c.fetchone()[0]
    print(f"\nTheorems with proofs available: {with_proofs:,}")
    
    # Sample recent additions
    print("\nSample Recent Theorems:")
    c.execute("SELECT name, year, source FROM theorems ORDER BY id DESC LIMIT 10")
    for name, year, source in c.fetchall():
        year_str = str(year) if year else 'Unknown'
        print(f"  • {name[:60]:60} ({year_str:8}) - {source}")
    
    # Year distribution
    print("\nYear Distribution:")
    c.execute("""
        SELECT 
            CASE 
                WHEN year < 0 THEN 'Ancient (BC)'
                WHEN year < 1000 THEN '0-999 AD'
                WHEN year < 1500 THEN '1000-1499'
                WHEN year < 1700 THEN '1500-1699'
                WHEN year < 1800 THEN '1700-1799'
                WHEN year < 1900 THEN '1800-1899'
                WHEN year < 1950 THEN '1900-1949'
                WHEN year < 2000 THEN '1950-1999'
                WHEN year >= 2000 THEN '2000-present'
                ELSE 'Unknown'
            END as period,
            COUNT(*) as cnt
        FROM theorems
        GROUP BY period
        ORDER BY 
            CASE period
                WHEN 'Ancient (BC)' THEN 1
                WHEN '0-999 AD' THEN 2
                WHEN '1000-1499' THEN 3
                WHEN '1500-1699' THEN 4
                WHEN '1700-1799' THEN 5
                WHEN '1800-1899' THEN 6
                WHEN '1900-1949' THEN 7
                WHEN '1950-1999' THEN 8
                WHEN '2000-present' THEN 9
                ELSE 10
            END
    """)
    for period, count in c.fetchall():
        print(f"  {period:20} {count:8,}")
    
    print("\n" + "="*60)

# Main execution
def main():
    """Main comprehensive scraping execution"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MATHEMATICAL THEOREM SCRAPER")
    print("Target: Thousands to Hundreds of Thousands of Theorems")
    print("="*60)
    print(f"\nStarting at {datetime.now()}")
    
    # Initialize database
    conn = init_db('theorems_comprehensive.db')
    
    # Run parallel scraping
    total_inserted = parallel_scraping(conn)
    
    print(f"\nTotal new theorems inserted: {total_inserted:,}")
    
    # Display comprehensive statistics
    display_comprehensive_stats(conn)
    
    # Close database
    conn.close()
    
    print(f"\nScraping completed at {datetime.now()}")
    print("Database saved as: theorems_comprehensive.db")
    print("\nTo query the database:")
    print("  sqlite3 theorems_comprehensive.db")
    print("  SELECT COUNT(*) FROM theorems;")
    print("  SELECT * FROM theorems WHERE name LIKE '%Pythagorean%';")

if __name__ == '__main__':
    main()