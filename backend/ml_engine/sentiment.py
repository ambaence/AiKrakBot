from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import logging
import time
from config import NEWSAPI_KEY, NEWS_CACHE_TTL

# Setup logging
logging.basicConfig(filename='logs/bot.log', level=logging.INFO)

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer with VADER, NewsAPI, and caching."""
        self.analyzer = SentimentIntensityAnalyzer()
        self.api_key = NEWSAPI_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        self.keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto"]
        self.logger = logging.getLogger(__name__)
        self.cache = {'sentiment': 0.0, 'timestamp': 0}  # In-memory cache

    def fetch_news(self, count=10):
        """Fetch recent news articles from NewsAPI."""
        if not self.api_key:
            self.logger.error("NewsAPI key not provided in .env")
            return []

        params = {
            'q': ' OR '.join(self.keywords),
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': count
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            news_texts = [article['title'] + " " + (article.get('description') or '') for article in articles]
            self.logger.info(f"Fetched {len(news_texts)} news articles from NewsAPI")
            return news_texts
        except requests.RequestException as e:
            self.logger.error(f"Error fetching news: {e}")
            return []

    def analyze(self, text):
        """Analyze sentiment of a given text using VADER."""
        try:
            score = self.analyzer.polarity_scores(text)['compound']
            return score  # -1 (negative) to 1 (positive)
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return 0.0

    def get_sentiment(self):
        """Fetch and analyze sentiment with caching."""
        current_time = time.time()
        cache_age = current_time - self.cache['timestamp']

        if cache_age < NEWS_CACHE_TTL and self.cache['sentiment'] != 0.0:
            self.logger.info(f"Using cached news sentiment: {self.cache['sentiment']:.2f} (age: {cache_age:.0f}s)")
            return self.cache['sentiment']

        news_texts = self.fetch_news()
        if not news_texts:
            self.logger.warning("No news articles retrieved; using neutral sentiment")
            self.cache['sentiment'] = 0.0
            self.cache['timestamp'] = current_time
            return 0.0

        sentiment_scores = [self.analyze(text) for text in news_texts]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        self.logger.info(f"New average news sentiment score: {avg_sentiment:.2f} from {len(news_texts)} articles")
        
        self.cache['sentiment'] = avg_sentiment
        self.cache['timestamp'] = current_time
        return avg_sentiment