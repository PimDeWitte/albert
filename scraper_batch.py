#!/usr/bin/env python3
"""
Optimized Batch Theorem Scraper
Stable, incremental scraping for 100,000+ theorems
"""

import sqlite3
import requests
from bs4 import BeautifulSoup
import arxiv
import time
import re
import os
from datetime import datetime
import json
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = 'theorems_ultimate.db'
BATCH_SIZE = 100
MAX_WORKERS = 4  # Conservative for stability
REQUEST_DELAY = 1  # Seconds between requests

class TheoremScraper:
    def __init__(self):
        self.conn = self.init_db()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def init_db(self):
        """Initialize database with progress tracking"""
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        
        c = conn.cursor()
        # Main theorems table
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
                era TEXT,
                source TEXT,
                source_url TEXT,
                category TEXT,
                citations INTEGER,
                hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Progress tracking table
        c.execute('''
            CREATE TABLE IF NOT EXISTS scraping_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                category TEXT,
                last_page INTEGER DEFAULT 0,
                last_offset INTEGER DEFAULT 0,
                completed BOOLEAN DEFAULT 0,
                total_found INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('CREATE INDEX IF NOT EXISTS idx_hash ON theorems(hash)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_source ON theorems(source)')
        conn.commit()
        return conn
    
    def get_era(self, year):
        """Categorize by era"""
        if year is None:
            return 'unknown'
        if year < 0:
            return 'ancient'
        elif year < 500:
            return 'classical'
        elif year < 1500:
            return 'medieval'
        elif year < 1800:
            return 'early-modern'
        elif year < 1900:
            return '19th-century'
        elif year < 2000:
            return '20th-century'
        else:
            return '21st-century'
    
    def extract_year(self, text):
        """Extract year from text"""
        if not text:
            return None
        
        # BC dates
        bc_match = re.search(r'(\d+)\s*BC', text, re.IGNORECASE)
        if bc_match:
            return -int(bc_match.group(1))
        
        # Standard years
        year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
        return int(year_match.group()) if year_match else None
    
    def generate_hash(self, name, source):
        """Generate unique hash"""
        content = f"{name}_{source}".lower().strip()
        return hashlib.md5(content.encode()).hexdigest()
    
    def insert_theorem(self, theorem):
        """Insert single theorem"""
        c = self.conn.cursor()
        theorem_hash = self.generate_hash(theorem['name'], theorem['source'])
        
        try:
            c.execute('''
                INSERT OR IGNORE INTO theorems 
                (name, year, description, formula, proof_available, authors, 
                 field, era, source, source_url, category, citations, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                theorem.get('name')[:500],
                theorem.get('year'),
                theorem.get('desc', theorem.get('description', ''))[:1000],
                theorem.get('formula', '')[:500],
                theorem.get('proof_available', 0),
                theorem.get('authors', '')[:300],
                theorem.get('field', '')[:100],
                self.get_era(theorem.get('year')),
                theorem.get('source'),
                theorem.get('source_url', '')[:500],
                theorem.get('category', '')[:100],
                theorem.get('citations', 0),
                theorem_hash
            ))
            self.conn.commit()
            return c.rowcount > 0
        except Exception as e:
            logger.error(f"Error inserting theorem: {e}")
            return False
    
    def get_progress(self, source, category=''):
        """Get scraping progress"""
        c = self.conn.cursor()
        c.execute(
            'SELECT last_page, last_offset, completed FROM scraping_progress WHERE source=? AND category=?',
            (source, category)
        )
        result = c.fetchone()
        return result if result else (0, 0, False)
    
    def update_progress(self, source, category='', page=0, offset=0, completed=False, total=0):
        """Update scraping progress"""
        c = self.conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO scraping_progress 
            (source, category, last_page, last_offset, completed, total_found, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (source, category, page, offset, completed, total))
        self.conn.commit()
    
    def scrape_arxiv_batch(self, category='math.AC', batch_size=100):
        """Scrape arXiv in batches with progress tracking"""
        logger.info(f"Starting arXiv batch scraping for {category}")
        
        last_page, last_offset, completed = self.get_progress('arXiv', category)
        if completed:
            logger.info(f"arXiv {category} already completed")
            return 0
        
        client = arxiv.Client()
        theorems_found = 0
        
        search_terms = ['theorem', 'lemma', 'corollary', 'proposition', 'proof']
        
        for term in search_terms:
            try:
                query = f'cat:{category} AND (ti:{term} OR abs:{term})'
                search = arxiv.Search(
                    query=query,
                    max_results=2000,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                batch_count = 0
                for result in client.results(search):
                    theorem = {
                        'name': result.title,
                        'year': result.published.year if result.published else None,
                        'desc': result.summary[:800],
                        'authors': ', '.join([author.name for author in result.authors])[:300],
                        'field': category,
                        'source': 'arXiv',
                        'source_url': result.entry_id,
                        'category': 'research'
                    }
                    
                    if self.insert_theorem(theorem):
                        theorems_found += 1
                        batch_count += 1
                        
                        if batch_count % batch_size == 0:
                            self.update_progress('arXiv', category, total=theorems_found)
                            logger.info(f"arXiv {category}: {theorems_found} theorems so far")
                            time.sleep(REQUEST_DELAY)
                
                time.sleep(2)  # Rate limiting between terms
                
            except Exception as e:
                logger.error(f"Error scraping arXiv {category} for {term}: {e}")
                time.sleep(5)
        
        self.update_progress('arXiv', category, completed=True, total=theorems_found)
        logger.info(f"Completed arXiv {category}: {theorems_found} theorems")
        return theorems_found
    
    def scrape_wikipedia_batch(self):
        """Scrape Wikipedia mathematics pages in batches"""
        logger.info("Starting Wikipedia batch scraping")
        
        last_page, last_offset, completed = self.get_progress('Wikipedia', 'math_categories')
        if completed:
            logger.info("Wikipedia already completed")
            return 0
        
        theorems_found = 0
        
        # Wikipedia API for mathematics categories
        api_url = 'https://en.wikipedia.org/api/rest_v1/page/related/'
        
        # Key mathematics pages to start from
        seed_pages = [
            'Mathematical_theorem', 'List_of_theorems', 'Fundamental_theorem',
            'Mathematical_proof', 'List_of_lemmas', 'Mathematical_conjecture'
        ]
        
        for page in seed_pages:
            try:
                url = f'{api_url}{page}'
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    pages = data.get('pages', [])
                    
                    for related_page in pages[:50]:  # Limit per seed
                        title = related_page.get('title', '')
                        extract = related_page.get('extract', '')
                        
                        if any(kw in title.lower() or kw in extract.lower() 
                               for kw in ['theorem', 'lemma', 'proof', 'formula', 'law', 'principle']):
                            
                            theorem = {
                                'name': title,
                                'desc': extract[:800],
                                'year': self.extract_year(extract),
                                'source': 'Wikipedia',
                                'source_url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                'category': 'encyclopedia'
                            }
                            
                            if self.insert_theorem(theorem):
                                theorems_found += 1
                
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Error scraping Wikipedia page {page}: {e}")
        
        self.update_progress('Wikipedia', 'math_categories', completed=True, total=theorems_found)
        logger.info(f"Completed Wikipedia: {theorems_found} theorems")
        return theorems_found
    
    def scrape_proofwiki_batch(self):
        """Scrape ProofWiki in batches"""
        logger.info("Starting ProofWiki batch scraping")
        
        last_page, last_offset, completed = self.get_progress('ProofWiki')
        if completed:
            logger.info("ProofWiki already completed")
            return 0
        
        theorems_found = 0
        base_url = 'https://proofwiki.org'
        
        # Categories to scrape
        categories = [
            'Category:Theorems', 'Category:Lemmas', 'Category:Corollaries',
            'Category:Propositions', 'Category:Definitions'
        ]
        
        for category in categories:
            try:
                # Get category pages
                url = f'{base_url}/wiki/{category}'
                response = self.session.get(url, timeout=30)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all page links
                content = soup.find('div', {'id': 'mw-pages'})
                if content:
                    for link in content.find_all('a'):
                        title = link.text.strip()
                        href = link.get('href', '')
                        
                        if href.startswith('/wiki/') and ':' not in href:
                            theorem = {
                                'name': title,
                                'desc': f"Mathematical theorem from ProofWiki: {title}",
                                'proof_available': 1,
                                'source': 'ProofWiki',
                                'source_url': base_url + href,
                                'category': category.replace('Category:', '').lower()
                            }
                            
                            if self.insert_theorem(theorem):
                                theorems_found += 1
                
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Error scraping ProofWiki {category}: {e}")
        
        self.update_progress('ProofWiki', completed=True, total=theorems_found)
        logger.info(f"Completed ProofWiki: {theorems_found} theorems")
        return theorems_found
    
    def scrape_mathworld_batch(self):
        """Scrape MathWorld in batches"""
        logger.info("Starting MathWorld batch scraping")
        
        last_page, last_offset, completed = self.get_progress('MathWorld')
        if completed:
            logger.info("MathWorld already completed")
            return 0
        
        theorems_found = 0
        base_url = 'https://mathworld.wolfram.com'
        
        # Alphabet-based scraping
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            try:
                url = f'{base_url}/letters/{letter}.html'
                response = self.session.get(url, timeout=30)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for link in soup.find_all('a'):
                    text = link.text.strip()
                    href = link.get('href', '')
                    
                    if (href and not href.startswith('http') and
                        any(kw in text.lower() for kw in 
                            ['theorem', 'lemma', 'law', 'principle', 'rule',
                             'formula', 'identity', 'inequality'])):
                        
                        theorem = {
                            'name': text,
                            'desc': f"Mathematical concept from MathWorld: {text}",
                            'source': 'MathWorld',
                            'source_url': base_url + '/' + href,
                            'category': 'reference'
                        }
                        
                        if self.insert_theorem(theorem):
                            theorems_found += 1
                
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Error scraping MathWorld letter {letter}: {e}")
        
        self.update_progress('MathWorld', completed=True, total=theorems_found)
        logger.info(f"Completed MathWorld: {theorems_found} theorems")
        return theorems_found
    
    def scrape_oeis_batch(self):
        """Scrape OEIS sequences in batches"""
        logger.info("Starting OEIS batch scraping")
        
        last_page, last_offset, completed = self.get_progress('OEIS')
        if completed:
            logger.info("OEIS already completed")
            return 0
        
        theorems_found = 0
        base_url = 'https://oeis.org'
        
        # Search for theorem-related sequences
        keywords = ['theorem', 'formula', 'identity', 'conjecture', 'lemma']
        
        for keyword in keywords:
            try:
                search_url = f'{base_url}/search'
                params = {'q': keyword, 'fmt': 'json'}
                response = self.session.get(search_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    for result in results[:200]:  # Limit per keyword
                        if isinstance(result, dict) and 'name' in result:
                            name = result['name']
                            if any(kw in name.lower() for kw in ['theorem', 'formula']):
                                seq_num = result.get('number', '')
                                
                                theorem = {
                                    'name': f"OEIS {seq_num}: {name}",
                                    'desc': result.get('comment', '')[:500],
                                    'formula': result.get('formula', ''),
                                    'source': 'OEIS',
                                    'source_url': f"{base_url}/{seq_num}",
                                    'category': 'sequences'
                                }
                                
                                if self.insert_theorem(theorem):
                                    theorems_found += 1
                
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Error scraping OEIS for {keyword}: {e}")
        
        self.update_progress('OEIS', completed=True, total=theorems_found)
        logger.info(f"Completed OEIS: {theorems_found} theorems")
        return theorems_found
    
    def run_batch_cycle(self):
        """Run one complete batch cycle"""
        logger.info("Starting batch scraping cycle")
        
        total_found = 0
        
        # arXiv categories
        arxiv_categories = [
            'math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA',
            'math.CO', 'math.CT', 'math.CV', 'math.DG', 'math.DS',
            'math.FA', 'math.GM', 'math.GN', 'math.GR', 'math.GT',
            'math.HO', 'math.IT', 'math.KT', 'math.LO', 'math.MG',
            'math.MP', 'math.NA', 'math.NT', 'math.OA', 'math.OC',
            'math.PR', 'math.QA', 'math.RA', 'math.RT', 'math.SG',
            'math.SP', 'math.ST'
        ]
        
        # Scrape arXiv categories
        for category in arxiv_categories:
            try:
                found = self.scrape_arxiv_batch(category)
                total_found += found
                logger.info(f"Progress: {total_found} total theorems collected")
                
                # Status check
                c = self.conn.cursor()
                c.execute("SELECT COUNT(*) FROM theorems")
                current_total = c.fetchone()[0]
                logger.info(f"Database now contains: {current_total:,} theorems")
                
                if current_total >= 100000:
                    logger.info("🎯 REACHED 100,000 THEOREMS!")
                    break
                    
            except Exception as e:
                logger.error(f"Error in arXiv batch {category}: {e}")
                continue
        
        # Other sources
        sources = [
            self.scrape_wikipedia_batch,
            self.scrape_proofwiki_batch,
            self.scrape_mathworld_batch,
            self.scrape_oeis_batch
        ]
        
        for scraper in sources:
            try:
                found = scraper()
                total_found += found
                
                c = self.conn.cursor()
                c.execute("SELECT COUNT(*) FROM theorems")
                current_total = c.fetchone()[0]
                logger.info(f"Database now contains: {current_total:,} theorems")
                
                if current_total >= 100000:
                    logger.info("🎯 REACHED 100,000 THEOREMS!")
                    break
                    
            except Exception as e:
                logger.error(f"Error in source scraper: {e}")
                continue
        
        return total_found
    
    def show_stats(self):
        """Display comprehensive statistics"""
        c = self.conn.cursor()
        
        # Total count
        c.execute("SELECT COUNT(*) FROM theorems")
        total = c.fetchone()[0]
        
        print(f"\n{'='*60}")
        print(f"BATCH SCRAPER STATISTICS")
        print(f"{'='*60}")
        print(f"\n🎯 Total Theorems: {total:,}")
        
        if total == 0:
            print("No theorems collected yet.")
            return
        
        # By source
        print(f"\n📚 By Source:")
        c.execute("SELECT source, COUNT(*) as cnt FROM theorems GROUP BY source ORDER BY cnt DESC")
        for source, count in c.fetchall():
            print(f"  {source:20} {count:8,}")
        
        # By era
        print(f"\n📅 By Era:")
        c.execute("SELECT era, COUNT(*) as cnt FROM theorems GROUP BY era ORDER BY cnt DESC")
        for era, count in c.fetchall():
            print(f"  {era:20} {count:8,}")
        
        # Progress by source
        print(f"\n⚡ Scraping Progress:")
        c.execute("SELECT source, category, completed, total_found FROM scraping_progress ORDER BY updated_at DESC")
        for source, category, completed, found in c.fetchall():
            status = "✅ Complete" if completed else "🔄 In Progress"
            cat_str = f"/{category}" if category else ""
            print(f"  {source}{cat_str:25} {found:6,} theorems {status}")
        
        # Recent additions
        print(f"\n🆕 Recent Additions:")
        c.execute("SELECT name, source FROM theorems ORDER BY id DESC LIMIT 5")
        for name, source in c.fetchall():
            print(f"  • {name[:50]:50} ({source})")
        
        print(f"\n{'='*60}")

def main():
    """Main batch scraping execution"""
    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZED BATCH THEOREM SCRAPER")
    print(f"🎯 Target: 100,000+ Theorems")
    print(f"⚡ Stable incremental collection")
    print(f"{'='*60}")
    
    scraper = TheoremScraper()
    
    # Show initial stats
    scraper.show_stats()
    
    try:
        # Run batch cycle
        print(f"\n🔥 Starting batch collection cycle...")
        total_collected = scraper.run_batch_cycle()
        
        print(f"\n✨ Batch cycle completed!")
        print(f"📈 New theorems collected this cycle: {total_collected:,}")
        
        # Final stats
        scraper.show_stats()
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
    finally:
        scraper.conn.close()
        print(f"\n✅ Database closed safely")
        print(f"📊 Results saved in: {DB_PATH}")

if __name__ == '__main__':
    main()