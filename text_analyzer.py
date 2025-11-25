"""
Text analysis module for document understanding.
Analyzes vocabulary, frequent terms, and co-occurrence patterns.
"""

import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextAnalyzer:
    """Analyzes text for vocabulary, frequency, and co-occurrence patterns."""
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the text analyzer.
        
        Args:
            language: Language for stopwords (default: 'english')
        """
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set()
            logger.warning(f"Stopwords for {language} not available, using empty set")
        
        # Add common document-specific stopwords
        self.stop_words.update(['invoice', 'receipt', 'date', 'total', 'subtotal', 
                               'tax', 'amount', 'due', 'number', 'item', 'items'])
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Preprocess text: tokenize, lowercase, remove punctuation.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def analyze_vocabulary(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze vocabulary across multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with vocabulary statistics
        """
        all_tokens = []
        unique_tokens = set()
        token_counts = Counter()
        
        for text in texts:
            tokens = self.preprocess_text(text, remove_stopwords=False)
            all_tokens.extend(tokens)
            unique_tokens.update(tokens)
            token_counts.update(tokens)
        
        # Calculate statistics
        total_tokens = len(all_tokens)
        unique_count = len(unique_tokens)
        avg_tokens_per_doc = total_tokens / len(texts) if texts else 0
        
        # Most frequent tokens (excluding stopwords)
        filtered_counts = {k: v for k, v in token_counts.items() 
                          if k.lower() not in self.stop_words and len(k) > 2}
        most_frequent = Counter(filtered_counts).most_common(50)
        
        return {
            'total_tokens': total_tokens,
            'unique_tokens': unique_count,
            'vocabulary_size': unique_count,
            'avg_tokens_per_document': avg_tokens_per_doc,
            'most_frequent_terms': most_frequent,
            'token_distribution': dict(token_counts.most_common(100))
        }
    
    def analyze_term_frequency(self, texts: List[str], top_n: int = 30) -> pd.DataFrame:
        """
        Analyze term frequency using TF-IDF.
        
        Args:
            texts: List of text strings
            top_n: Number of top terms to return
            
        Returns:
            DataFrame with term frequencies
        """
        # Use CountVectorizer for term frequency
        vectorizer = CountVectorizer(
            max_features=top_n * 2,
            stop_words=list(self.stop_words),
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Words with at least 3 characters
        )
        
        try:
            tf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate term frequencies
            tf_scores = np.array(tf_matrix.sum(axis=0)).flatten()
            
            # Create DataFrame
            df = pd.DataFrame({
                'term': feature_names,
                'frequency': tf_scores,
                'document_frequency': np.array((tf_matrix > 0).sum(axis=0)).flatten()
            })
            
            df = df.sort_values('frequency', ascending=False).head(top_n)
            df['relative_frequency'] = df['frequency'] / df['frequency'].sum()
            
            return df
        except Exception as e:
            logger.error(f"Error in term frequency analysis: {e}")
            return pd.DataFrame()
    
    def analyze_cooccurrence(self, texts: List[str], window_size: int = 3, top_n: int = 20) -> Dict[str, List[Tuple[str, int]]]:
        """
        Analyze co-occurrence patterns of terms.
        
        Args:
            texts: List of text strings
            window_size: Size of context window for co-occurrence
            top_n: Number of top co-occurrences per term
            
        Returns:
            Dictionary mapping terms to their top co-occurring terms
        """
        cooccurrence = defaultdict(lambda: defaultdict(int))
        all_terms = set()
        
        for text in texts:
            tokens = self.preprocess_text(text, remove_stopwords=True)
            all_terms.update(tokens)
            
            # Create n-grams for co-occurrence
            for i, token in enumerate(tokens):
                # Look at surrounding tokens within window
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        cooccurrence[token][tokens[j]] += 1
        
        # Get top co-occurrences for each term
        top_cooccurrences = {}
        for term in sorted(all_terms):
            if term in cooccurrence:
                cooc_list = sorted(cooccurrence[term].items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:top_n]
                if cooc_list:
                    top_cooccurrences[term] = cooc_list
        
        return top_cooccurrences
    
    def analyze_bigrams(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Analyze bigram (2-word) phrases.
        
        Args:
            texts: List of text strings
            top_n: Number of top bigrams to return
            
        Returns:
            List of (bigram, count) tuples
        """
        all_bigrams = []
        
        for text in texts:
            tokens = self.preprocess_text(text, remove_stopwords=True)
            bigrams = list(ngrams(tokens, 2))
            all_bigrams.extend([' '.join(bg) for bg in bigrams])
        
        bigram_counts = Counter(all_bigrams)
        return bigram_counts.most_common(top_n)
    
    def analyze_trigrams(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Analyze trigram (3-word) phrases.
        
        Args:
            texts: List of text strings
            top_n: Number of top trigrams to return
            
        Returns:
            List of (trigram, count) tuples
        """
        all_trigrams = []
        
        for text in texts:
            tokens = self.preprocess_text(text, remove_stopwords=True)
            trigrams = list(ngrams(tokens, 3))
            all_trigrams.extend([' '.join(tg) for tg in trigrams])
        
        trigram_counts = Counter(all_trigrams)
        return trigram_counts.most_common(top_n)
    
    def analyze_topic_modeling(self, texts: List[str], n_topics: int = 5, top_words: int = 10) -> Dict[str, List[str]]:
        """
        Perform topic modeling using LDA.
        
        Args:
            texts: List of text strings
            n_topics: Number of topics to extract
            top_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic numbers to top words
        """
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words=list(self.stop_words),
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(doc_term_matrix)
            
            # Extract top words for each topic
            topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-top_words:][::-1]
                top_words_list = [feature_names[i] for i in top_indices]
                topics[f'Topic {topic_idx + 1}'] = top_words_list
            
            return topics
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return {}
    
    def generate_analysis_report(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Analyzing vocabulary...")
        vocabulary = self.analyze_vocabulary(texts)
        
        logger.info("Analyzing term frequency...")
        term_freq = self.analyze_term_frequency(texts)
        
        logger.info("Analyzing co-occurrence patterns...")
        cooccurrence = self.analyze_cooccurrence(texts)
        
        logger.info("Analyzing bigrams...")
        bigrams = self.analyze_bigrams(texts)
        
        logger.info("Analyzing trigrams...")
        trigrams = self.analyze_trigrams(texts)
        
        logger.info("Performing topic modeling...")
        topics = self.analyze_topic_modeling(texts)
        
        return {
            'vocabulary': vocabulary,
            'term_frequency': term_freq.to_dict('records') if not term_freq.empty else [],
            'cooccurrence': cooccurrence,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'topics': topics
        }


if __name__ == "__main__":
    # Example usage
    analyzer = TextAnalyzer()
    
    sample_texts = [
        "Invoice number INV-001 total amount due",
        "Receipt date payment method credit card",
        "Invoice total amount payment due date"
    ]
    
    report = analyzer.generate_analysis_report(sample_texts)
    print("Vocabulary analysis:")
    print(f"Total tokens: {report['vocabulary']['total_tokens']}")
    print(f"Unique tokens: {report['vocabulary']['unique_tokens']}")
    print("\nTop terms:")
    for term, count in report['vocabulary']['most_frequent_terms'][:10]:
        print(f"  {term}: {count}")

