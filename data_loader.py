"""
Data loading script for LayoutLM dataset.
Downloads and loads OCR text content from the dataset.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayoutLMDataLoader:
    """Loads OCR text data from LayoutLM dataset."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory where dataset will be stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir = self.data_dir / "raw"
        self.raw_data_dir.mkdir(exist_ok=True)
        
    def download_dataset(self, kaggle_dataset: str = None):
        """
        Download dataset from Kaggle.
        
        Note: Requires Kaggle API credentials in ~/.kaggle/kaggle.json
        If dataset is not available, creates sample data for demonstration.
        
        Args:
            kaggle_dataset: Kaggle dataset identifier (e.g., 'microsoft/layoutlm')
        """
        try:
            import kaggle
            if kaggle_dataset:
                logger.info(f"Downloading dataset: {kaggle_dataset}")
                kaggle.api.dataset_download_files(
                    kaggle_dataset,
                    path=str(self.raw_data_dir),
                    unzip=True
                )
                logger.info("Dataset downloaded successfully")
            else:
                logger.warning("No Kaggle dataset specified. Creating sample data.")
                self._create_sample_data()
        except Exception as e:
            logger.warning(f"Could not download from Kaggle: {e}")
            logger.info("Creating sample data for demonstration purposes.")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample OCR data for demonstration if dataset is not available."""
        sample_documents = [
            {
                "id": "doc_001",
                "text": "INVOICE #INV-2024-001\nDate: March 15, 2024\nCustomer: ABC Corporation\n123 Main Street\nNew York, NY 10001\n\nItem Description\tQuantity\tUnit Price\tTotal\nWidget A\t\t2\t\t$25.00\t\t$50.00\nWidget B\t\t1\t\t$30.00\t\t$30.00\n\nSubtotal: $80.00\nTax (8.5%): $6.80\nTotal: $86.80\n\nPayment Terms: Net 30\nDue Date: April 14, 2024"
            },
            {
                "id": "doc_002",
                "text": "RECEIPT\nReceipt #: RCP-2024-0423\nDate: April 23, 2024\nTime: 14:35\n\nMerchant: XYZ Store\n456 Commerce Blvd\nLos Angeles, CA 90001\n\nItems Purchased:\nProduct 1\t\t$12.99\nProduct 2\t\t$8.50\nProduct 3\t\t$24.99\n\nSubtotal: $46.48\nTax: $3.72\nTotal: $50.20\n\nPayment Method: Credit Card\nCard: ****1234\nTransaction ID: TXN-789456123"
            },
            {
                "id": "doc_003",
                "text": "INVOICE\nInvoice Number: INV-789456\nDate: May 1, 2024\nBill To:\nTech Solutions Inc.\n789 Tech Park\nSan Francisco, CA 94102\n\nServices Rendered:\nConsulting Hours (10 hrs)\t$150.00/hr\t$1,500.00\nDevelopment (20 hrs)\t$200.00/hr\t$4,000.00\n\nSubtotal: $5,500.00\nDiscount (10%): -$550.00\nTax (9.5%): $470.25\nTotal Amount Due: $5,420.25\n\nPayment Due: May 31, 2024\nPO Number: PO-2024-567"
            },
            {
                "id": "doc_004",
                "text": "RECEIPT\nReceipt #: RCP-2024-0515\nDate: May 15, 2024\n\nRestaurant: Fine Dining Co.\n321 Food Street\nChicago, IL 60601\n\nOrder Details:\nAppetizer\t\t$15.00\nMain Course\t\t$32.00\nDessert\t\t\t$12.00\nBeverages (2)\t\t$8.00\n\nSubtotal: $67.00\nService Charge (18%): $12.06\nTax (10%): $7.91\nTotal: $86.97\n\nTip: $15.00\nGrand Total: $101.97\n\nThank you for your visit!"
            },
            {
                "id": "doc_005",
                "text": "INVOICE\nInvoice #: INV-2024-100\nDate: June 10, 2024\n\nClient: Global Enterprises Ltd.\n555 Business Ave\nBoston, MA 02101\n\nDescription\t\tAmount\nMonthly Subscription\t$299.00\nSetup Fee\t\t$99.00\n\nSubtotal: $398.00\nTax (6.25%): $24.88\nTotal: $422.88\n\nPayment Terms: Due on Receipt\nInvoice Date: June 10, 2024\nDue Date: June 10, 2024\nAccount Number: ACC-789456"
            }
        ]
        
        # Save sample data as JSON files
        for doc in sample_documents:
            output_file = self.raw_data_dir / f"{doc['id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(sample_documents)} sample documents")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all documents from the data directory.
        
        Returns:
            List of document dictionaries with 'id' and 'text' keys
        """
        documents = []
        
        # Look for JSON files
        json_files = list(self.raw_data_dir.glob("*.json"))
        
        if not json_files:
            logger.warning("No JSON files found. Creating sample data.")
            self._create_sample_data()
            json_files = list(self.raw_data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def load_as_dataframe(self) -> pd.DataFrame:
        """
        Load documents as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: 'id', 'text', 'text_length'
        """
        documents = self.load_documents()
        
        df = pd.DataFrame(documents)
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        return df
    
    def get_all_text(self) -> str:
        """
        Get all text content concatenated.
        
        Returns:
            Combined text from all documents
        """
        documents = self.load_documents()
        return "\n\n".join([doc['text'] for doc in documents])


if __name__ == "__main__":
    # Example usage
    loader = LayoutLMDataLoader()
    loader.download_dataset()  # Will create sample data if Kaggle not available
    df = loader.load_as_dataframe()
    print(f"Loaded {len(df)} documents")
    print("\nFirst document preview:")
    print(df.iloc[0]['text'][:200])

