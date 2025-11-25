"""
Main script to orchestrate the document understanding pipeline.
"""

import argparse
import sys
import logging
from pathlib import Path

from data_loader import LayoutLMDataLoader
from ner_extractor import DocumentNER
from text_analyzer import TextAnalyzer
from report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Document Understanding Analysis Pipeline'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing the dataset (default: data)'
    )
    parser.add_argument(
        '--kaggle-dataset',
        type=str,
        default=None,
        help='Kaggle dataset identifier (e.g., microsoft/layoutlm)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='report.html',
        help='Output report file path (default: report.html)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        choices=['en', 'es', 'de'],
        help='Report language: en (English), es (Spanish), de (German) (default: en)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download (use existing data)'
    )
    
    args = parser.parse_args()
    
    try:
        # Step 1: Load data
        logger.info("=" * 60)
        logger.info("Step 1: Loading data")
        logger.info("=" * 60)
        
        loader = LayoutLMDataLoader(data_dir=args.data_dir)
        
        if not args.skip_download:
            loader.download_dataset(kaggle_dataset=args.kaggle_dataset)
        else:
            logger.info("Skipping dataset download (using existing data)")
        
        documents = loader.load_documents()
        
        if not documents:
            logger.error("No documents found. Please check your data directory.")
            sys.exit(1)
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Step 2: Extract entities
        logger.info("=" * 60)
        logger.info("Step 2: Extracting entities")
        logger.info("=" * 60)
        
        ner = DocumentNER()
        ner_results = ner.extract_from_documents(documents)
        ner_summary = ner.get_entity_summary(ner_results)
        
        logger.info(f"Extracted {ner_summary['total_entities']} entities")
        logger.info(f"Entity types found: {list(ner_summary['entity_type_counts'].keys())}")
        
        # Step 3: Analyze text
        logger.info("=" * 60)
        logger.info("Step 3: Analyzing text")
        logger.info("=" * 60)
        
        texts = [doc['text'] for doc in documents]
        analyzer = TextAnalyzer(language=args.language)
        text_analysis = analyzer.generate_analysis_report(texts)
        
        logger.info(f"Vocabulary size: {text_analysis['vocabulary']['vocabulary_size']}")
        logger.info(f"Total tokens: {text_analysis['vocabulary']['total_tokens']}")
        
        # Step 4: Generate report
        logger.info("=" * 60)
        logger.info("Step 4: Generating report")
        logger.info("=" * 60)
        
        report_gen = ReportGenerator(language=args.language)
        report_path = report_gen.generate_html_report(
            ner_results=ner_results,
            ner_summary=ner_summary,
            text_analysis=text_analysis,
            output_file=args.output
        )
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Report generated: {report_path}")
        logger.info(f"Open the report in your browser to view the results.")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Documents analyzed: {ner_summary['total_documents']}")
        print(f"Entities extracted: {ner_summary['total_entities']}")
        print(f"Vocabulary size: {text_analysis['vocabulary']['vocabulary_size']}")
        print(f"Total tokens: {text_analysis['vocabulary']['total_tokens']:,}")
        
        if ner_summary.get('monetary_totals'):
            monetary = ner_summary['monetary_totals']
            print(f"\nMonetary Analysis:")
            print(f"  Total: ${monetary['sum']:,.2f}")
            print(f"  Average: ${monetary['average']:,.2f}")
            print(f"  Range: ${monetary['min']:,.2f} - ${monetary['max']:,.2f}")
        
        print(f"\nReport saved to: {report_path}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

