# ============================================================================
# nlp_pipeline.py - YOUR JUPYTER NOTEBOOK CODE CONVERTED TO PYTHON MODULE
# ============================================================================
# This file contains all the scraping, sentiment analysis, and CSV generation
# logic extracted from your Jupyter notebook.
# ============================================================================

import os
import re
import time
import random
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import requests
    from bs4 import BeautifulSoup
    import cloudscraper
    import nltk
    nltk.download('vader_lexicon', quiet=True)
except ImportError:
    print("Warning: Some NLP packages not installed. Install with:")
    print("pip install vaderSentiment beautifulsoup4 cloudscraper lxml")


# ============================================================================
# CLASS 1: PRODUCT SCRAPER
# ============================================================================

class ProductScraper:
    """Advanced scraper for Amazon product reviews"""
    
    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        self.session = requests.Session()
        
        self.desktop_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        ]
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(text.split()).strip()
    
    def _extract_product_name(self, soup) -> str:
        selectors = ['#productTitle', 'h1#title', 'span#productTitle']
        for selector in selectors:
            try:
                elem = soup.select_one(selector)
                if elem:
                    text = self._clean_text(elem.get_text())
                    if text and len(text) > 5:
                        return text
            except:
                continue
        return "Unknown Product"
    
    def _extract_price(self, soup) -> float:
        price_selectors = [
            'span.a-price-whole',
            'span.a-price span.a-offscreen',
            '#priceblock_ourprice',
        ]
        for selector in price_selectors:
            try:
                elems = soup.select(selector)
                for elem in elems:
                    text = elem.get_text() if hasattr(elem, 'get_text') else elem.text
                    text = re.sub(r'[₹$,\s]', '', text)
                    match = re.search(r'(\d+\.?\d*)', text)
                    if match:
                        price = float(match.group(1))
                        if 100 < price < 1000000:
                            return price
            except:
                continue
        return 0.0
    
    def _extract_reviews_and_ratings(self, soup) -> Tuple[List[str], List[float]]:
        reviews = []
        ratings = []
        review_containers = ['[data-hook="review"]', '.review', '.a-section.review']
        
        for container_selector in review_containers:
            review_elements = soup.select(container_selector)
            for review_elem in review_elements:
                try:
                    text_selectors = ['[data-hook="review-body"]', '.review-text']
                    review_text = None
                    
                    for text_sel in text_selectors:
                        text_elem = review_elem.select_one(text_sel)
                        if text_elem:
                            review_text = self._clean_text(text_elem.get_text())
                            break
                    
                    if not review_text or len(review_text) < 20:
                        continue
                    
                    rating = 3.0
                    rating_elem = review_elem.select_one('[data-hook="review-star-rating"]')
                    if rating_elem:
                        rating_text = rating_elem.get_text()
                        match = re.search(r'(\d+\.?\d*)', rating_text)
                        if match:
                            rating = float(match.group(1))
                    
                    reviews.append(review_text)
                    ratings.append(rating)
                
                except Exception as e:
                    continue
            
            if reviews:
                break
        
        return reviews, ratings
    
    def scrape_product(self, url: str) -> Dict:
        print(f"▶ Scraping: {url[:60]}...")
        
        try:
            response = self.scraper.get(url, timeout=20)
            
            if response.status_code != 200:
                print(f"✗ Request failed with status {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            page_text = soup.get_text().lower()
            if 'captcha' in page_text or 'robot' in page_text[:500]:
                print("✗ Captcha detected or blocked")
                return None
            
            product_name = self._extract_product_name(soup)
            price = self._extract_price(soup)
            reviews, ratings = self._extract_reviews_and_ratings(soup)
            
            if reviews:
                print(f"✓ Successfully scraped {len(reviews)} reviews")
                return {
                    'product_name': product_name,
                    'price': price,
                    'reviews': reviews,
                    'ratings': ratings
                }
            else:
                print("✗ No reviews found")
                return None
        
        except Exception as e:
            print(f"✗ Scraping error: {e}")
            return None


# ============================================================================
# CLASS 2: SENTIMENT ANALYZER
# ============================================================================

class SentimentAnalyzer:
    """VADER-based sentiment analysis"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def predict(self, text: str) -> Dict:
        if not text:
            return {'sentiment': 'Neutral', 'compound_score': 0.0, 'pos_score': 0.0, 'neg_score': 0.0, 'neu_score': 1.0}
        
        scores = self.sia.polarity_scores(str(text))
        compound_score = scores['compound']
        
        if compound_score >= 0.05:
            sentiment_label = 'Positive'
        elif compound_score <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        return {
            'sentiment': sentiment_label,
            'compound_score': float(compound_score),
            'pos_score': float(scores['pos']),
            'neg_score': float(scores['neg']),
            'neu_score': float(scores['neu'])
        }


# ============================================================================
# CLASS 3: FEATURE DETECTOR
# ============================================================================

class FeatureMentionDetector:
    """Detect mentions of delivery, quality, value in reviews"""
    
    def __init__(self):
        self.keywords = {
            'delivery': ['delivery', 'shipping', 'packaging', 'arrival', 'dispatch'],
            'quality': ['quality', 'durable', 'sturdy', 'defect', 'material'],
            'value': ['value', 'worth', 'expensive', 'cheap', 'money']
        }
    
    def detect_feature(self, text: str, feature: str) -> int:
        if not text:
            return 0
        text_lower = str(text).lower()
        keywords = self.keywords.get(feature, [])
        return 1 if any(kw in text_lower for kw in keywords) else 0


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def generate_csv_from_url(product_url: str, output_dir: str) -> str:
    """
    MAIN FUNCTION: Takes a product URL and generates a 22-column CSV.
    
    Args:
        product_url: Amazon product URL
        output_dir: Directory to save the CSV
    
    Returns:
        Path to generated CSV file, or None if failed
    """
    
    print("\n" + "="*80)
    print("AMAZON PRODUCT SCRAPER & NLP PIPELINE")
    print("="*80)
    
    # Step 1: Scrape
    print("\n[STEP 1] Web Scraping...")
    scraper = ProductScraper()
    product_data = scraper.scrape_product(product_url)
    
    if not product_data or not product_data.get('reviews'):
        print("✗ Scraping failed")
        return None
    
    print(f"✓ Scraped {len(product_data['reviews'])} reviews")
    print(f"  Product: {product_data['product_name'][:60]}")
    print(f"  Price: ₹{product_data['price']:,.2f}")
    
    # Step 2: Sentiment Analysis & CSV Generation
    print("\n[STEP 2] Sentiment Analysis & CSV Generation...")
    sentiment_analyzer = SentimentAnalyzer()
    feature_detector = FeatureMentionDetector()
    
    rows = []
    review_counter = 1
    
    for idx, (review, rating) in enumerate(zip(product_data['reviews'], product_data['ratings'])):
        sentiment_data = sentiment_analyzer.predict(review)
        
        delivery_mentioned = feature_detector.detect_feature(review, 'delivery')
        quality_mentioned = feature_detector.detect_feature(review, 'quality')
        value_mentioned = feature_detector.detect_feature(review, 'value')
        
        original_price = product_data['price'] * 1.2 if product_data['price'] > 0 else 0
        discount_percentage = int(((original_price - product_data['price']) / original_price * 100)) if original_price > 0 else 0
        
        row = {
            'review_id': f"R{review_counter:06d}",
            'product_id': f"B0{review_counter:08d}",
            'product_name': product_data['product_name'],
            'review_text': review,
            'rating': int(rating),
            'review_date': datetime.now().strftime('%Y-%m-%d'),
            'reviewer_name': f"Reviewer_{review_counter}",
            'verified_purchase': 1,
            'helpful_votes': random.randint(0, 100),
            'total_votes': random.randint(50, 150),
            'sentiment_label': sentiment_data['sentiment'],
            'compound_score': round(sentiment_data['compound_score'], 4),
            'positive_score': round(sentiment_data['pos_score'], 4),
            'negative_score': round(sentiment_data['neg_score'], 4),
            'neutral_score': round(sentiment_data['neu_score'], 4),
            'product_price': round(product_data['price'], 2),
            'original_price': round(original_price, 2),
            'discount_percentage': discount_percentage,
            'platform': 'Amazon',
            'delivery_mentioned': delivery_mentioned,
            'quality_mentioned': quality_mentioned,
            'value_mentioned': value_mentioned
        }
        
        rows.append(row)
        review_counter += 1
    
    # Step 3: Save CSV
    print("\n[STEP 3] Saving CSV...")
    
    column_order = [
        'review_id', 'product_id', 'product_name', 'review_text', 'rating',
        'review_date', 'reviewer_name', 'verified_purchase', 'helpful_votes',
        'total_votes', 'sentiment_label', 'compound_score', 'positive_score',
        'negative_score', 'neutral_score', 'product_price', 'original_price',
        'discount_percentage', 'platform', 'delivery_mentioned', 'quality_mentioned',
        'value_mentioned'
    ]
    
    df = pd.DataFrame(rows)
    df = df[column_order]
    
    filename = f"amazon_reviews_{int(time.time())}.csv"
    filepath = os.path.join(output_dir, filename)
    
    try:
        df.to_csv(filepath, index=False)
        print(f"✓ CSV saved: {filename}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"  Path: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error saving CSV: {e}")
        return None


# ============================================================================
# BATCH PROCESSING (for multiple URLs)
# ============================================================================

def generate_csv_for_products(urls: List[str], output_dir: str) -> List[str]:
    """
    Generate CSVs for multiple product URLs.
    
    Args:
        urls: List of Amazon product URLs
        output_dir: Directory to save CSVs
    
    Returns:
        List of generated CSV file paths
    """
    generated_csvs = []
    
    for i, url in enumerate(urls):
        print(f"\n\n{'='*80}")
        print(f"PROCESSING URL {i+1}/{len(urls)}")
        print(f"{'='*80}")
        
        try:
            csv_path = generate_csv_from_url(url, output_dir)
            if csv_path:
                generated_csvs.append(csv_path)
        except Exception as e:
            print(f"✗ Error processing URL: {e}")
            continue
    
    print(f"\n\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully generated {len(generated_csvs)}/{len(urls)} CSVs")
    
    return generated_csvs
