"""
Named Entity Recognition (NER) for document understanding.
Extracts document-specific entities: dates, monetary amounts, invoice numbers, etc.
"""

import re
import spacy
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dateutil import parser as date_parser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentNER:
    """Extracts document-specific entities from text."""
    
    def __init__(self):
        """Initialize the NER extractor."""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Installing...")
            logger.info("Please run: python -m spacy download en_core_web_sm")
            # Create a minimal nlp object for basic tokenization
            self.nlp = None
        
        # Regex patterns for document-specific entities
        self.patterns = {
            'invoice_number': [
                r'INVOICE\s*#?\s*:?\s*([A-Z0-9\-]+)',
                r'Invoice\s*Number\s*:?\s*([A-Z0-9\-]+)',
                r'Invoice\s*#\s*:?\s*([A-Z0-9\-]+)',
                r'INV[-\s]?(\d+)',
                r'#\s*([A-Z]{2,4}[-]?\d+)',
            ],
            'receipt_number': [
                r'RECEIPT\s*#?\s*:?\s*([A-Z0-9\-]+)',
                r'Receipt\s*#\s*:?\s*([A-Z0-9\-]+)',
                r'RCP[-\s]?(\d+)',
            ],
            'transaction_id': [
                r'Transaction\s*ID\s*:?\s*([A-Z0-9\-]+)',
                r'TXN[-\s]?([A-Z0-9\-]+)',
                r'Transaction\s*#\s*:?\s*([A-Z0-9\-]+)',
            ],
            'po_number': [
                r'PO\s*Number\s*:?\s*([A-Z0-9\-]+)',
                r'PO\s*#\s*:?\s*([A-Z0-9\-]+)',
                r'Purchase\s*Order\s*:?\s*([A-Z0-9\-]+)',
            ],
            'account_number': [
                r'Account\s*Number\s*:?\s*([A-Z0-9\-]+)',
                r'Account\s*#\s*:?\s*([A-Z0-9\-]+)',
                r'ACC[-\s]?([A-Z0-9\-]+)',
            ],
            'monetary_amount': [
                r'\$[\d,]+\.?\d*',
                r'USD\s*[\d,]+\.?\d*',
                r'[\d,]+\.?\d*\s*USD',
                r'Total\s*:?\s*\$?([\d,]+\.?\d*)',
                r'Amount\s*:?\s*\$?([\d,]+\.?\d*)',
            ],
            'date': [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
                r'Date\s*:?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
                r'Due\s*Date\s*:?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            'phone': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            ],
            'address': [
                r'\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Parkway|Pkwy)[,\s]+[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}',
            ],
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all document-specific entities from text.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary mapping entity types to lists of extracted entities
        """
        entities = {
            'invoice_number': [],
            'receipt_number': [],
            'transaction_id': [],
            'po_number': [],
            'account_number': [],
            'monetary_amount': [],
            'date': [],
            'email': [],
            'phone': [],
            'address': [],
        }
        
        # Extract using regex patterns
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)
                    entities[entity_type].append({
                        'text': value,
                        'start': match.start(),
                        'end': match.end(),
                        'full_match': match.group(0)
                    })
        
        # Remove duplicates (same text and position)
        for entity_type in entities:
            seen = set()
            unique_entities = []
            for entity in entities[entity_type]:
                key = (entity['text'], entity['start'], entity['end'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            entities[entity_type] = unique_entities
        
        # Use spaCy for additional entity extraction if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'MONEY':
                    entities['monetary_amount'].append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'full_match': ent.text,
                        'source': 'spacy'
                    })
                elif ent.label_ == 'DATE':
                    entities['date'].append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'full_match': ent.text,
                        'source': 'spacy'
                    })
        
        # Parse and normalize dates
        entities['date'] = self._normalize_dates(entities['date'])
        
        # Parse and normalize monetary amounts
        entities['monetary_amount'] = self._normalize_amounts(entities['monetary_amount'])
        
        return entities
    
    def _normalize_dates(self, date_entities: List[Dict]) -> List[Dict]:
        """Normalize date entities to standard format."""
        normalized = []
        for entity in date_entities:
            try:
                # Try to parse the date
                date_str = entity['text']
                parsed_date = date_parser.parse(date_str, fuzzy=True)
                entity['parsed'] = parsed_date.isoformat()
                entity['formatted'] = parsed_date.strftime('%Y-%m-%d')
            except:
                entity['parsed'] = None
                entity['formatted'] = entity['text']
            normalized.append(entity)
        return normalized
    
    def _normalize_amounts(self, amount_entities: List[Dict]) -> List[Dict]:
        """Normalize monetary amounts to numeric values."""
        normalized = []
        for entity in amount_entities:
            try:
                # Extract numeric value
                text = entity['text']
                # Remove currency symbols and commas
                numeric_str = re.sub(r'[^\d.]', '', text)
                if numeric_str:
                    entity['value'] = float(numeric_str)
                    entity['formatted'] = f"${entity['value']:,.2f}"
                else:
                    entity['value'] = None
                    entity['formatted'] = text
            except:
                entity['value'] = None
                entity['formatted'] = entity['text']
            normalized.append(entity)
        return normalized
    
    def extract_from_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple documents.
        
        Args:
            documents: List of document dictionaries with 'id' and 'text' keys
            
        Returns:
            List of dictionaries with document id and extracted entities
        """
        results = []
        for doc in documents:
            entities = self.extract_entities(doc['text'])
            results.append({
                'document_id': doc['id'],
                'entities': entities,
                'entity_counts': {k: len(v) for k, v in entities.items()}
            })
        return results
    
    def get_entity_summary(self, extraction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics of extracted entities.
        
        Args:
            extraction_results: Results from extract_from_documents
            
        Returns:
            Summary dictionary with statistics
        """
        summary = {
            'total_documents': len(extraction_results),
            'entity_type_counts': {},
            'total_entities': 0,
            'monetary_totals': [],
            'date_range': None,
        }
        
        all_amounts = []
        all_dates = []
        
        for result in extraction_results:
            for entity_type, entities in result['entities'].items():
                if entity_type not in summary['entity_type_counts']:
                    summary['entity_type_counts'][entity_type] = 0
                summary['entity_type_counts'][entity_type] += len(entities)
                summary['total_entities'] += len(entities)
                
                if entity_type == 'monetary_amount':
                    for entity in entities:
                        if 'value' in entity and entity['value'] is not None:
                            all_amounts.append(entity['value'])
                
                if entity_type == 'date':
                    for entity in entities:
                        if 'parsed' in entity and entity['parsed']:
                            try:
                                all_dates.append(datetime.fromisoformat(entity['parsed']))
                            except:
                                pass
        
        if all_amounts:
            summary['monetary_totals'] = {
                'sum': sum(all_amounts),
                'average': sum(all_amounts) / len(all_amounts),
                'min': min(all_amounts),
                'max': max(all_amounts),
                'count': len(all_amounts)
            }
        
        if all_dates:
            summary['date_range'] = {
                'earliest': min(all_dates).isoformat(),
                'latest': max(all_dates).isoformat(),
                'span_days': (max(all_dates) - min(all_dates)).days
            }
        
        return summary


if __name__ == "__main__":
    # Example usage
    ner = DocumentNER()
    
    sample_text = """
    INVOICE #INV-2024-001
    Date: March 15, 2024
    Customer: ABC Corporation
    123 Main Street, New York, NY 10001
    
    Total: $86.80
    Due Date: April 14, 2024
    """
    
    entities = ner.extract_entities(sample_text)
    print("Extracted entities:")
    for entity_type, values in entities.items():
        if values:
            print(f"\n{entity_type}:")
            for val in values:
                print(f"  - {val}")

