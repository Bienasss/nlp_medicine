# Document Understanding with NLP

This project applies Natural Language Processing (NLP) techniques to document understanding using the LayoutLM dataset. It extracts document-specific entities, analyzes text patterns, and generates comprehensive visual reports.

## Features

- **Data Loading**: Downloads and loads OCR text data from LayoutLM dataset (or uses sample data)
- **Named Entity Recognition (NER)**: Extracts document-specific entities:
  - Invoice/Receipt Numbers
  - Transaction IDs
  - PO Numbers
  - Account Numbers
  - Monetary Amounts
  - Dates
  - Email addresses
  - Phone numbers
  - Addresses
- **Text Analysis**:
  - Vocabulary statistics
  - Term frequency analysis (TF-IDF)
  - Co-occurrence patterns
  - Bigram and trigram analysis
  - Topic modeling (LDA)
- **Report Generation**: Beautiful HTML reports with visualizations
- **Multilingual Support**: Reports available in English, Spanish, and German

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd /home/benas/Projects/python/nlp_medicine
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

4. (Optional) Setup Kaggle API for dataset download:
   - Create a Kaggle account at https://www.kaggle.com
   - Go to Account settings and create an API token
   - Place `kaggle.json` in `~/.kaggle/` directory
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
python main.py
```

This will:
1. Create sample data (if no dataset is available)
2. Extract entities from documents
3. Perform text analysis
4. Generate an HTML report (`report.html`)

### Advanced Usage

#### Specify Output File

```bash
python main.py --output my_report.html
```

#### Change Report Language

```bash
# English (default)
python main.py --language en

# Spanish
python main.py --language es

# German
python main.py --language de
```

#### Download from Kaggle

```bash
python main.py --kaggle-dataset microsoft/layoutlm
```

#### Use Existing Data

```bash
python main.py --skip-download
```

#### Custom Data Directory

```bash
python main.py --data-dir /path/to/your/data
```

### Command Line Options

```
--data-dir DIR          Directory containing the dataset (default: data)
--kaggle-dataset ID     Kaggle dataset identifier
--output FILE           Output report file path (default: report.html)
--language LANG         Report language: en, es, or de (default: en)
--skip-download         Skip dataset download (use existing data)
```

## Project Structure

```
nlp_medicine/
├── main.py                 # Main orchestration script
├── data_loader.py          # Dataset loading module
├── ner_extractor.py        # Named Entity Recognition module
├── text_analyzer.py        # Text analysis module
├── report_generator.py     # HTML report generation module
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── data/                  # Data directory (created automatically)
    └── raw/               # Raw dataset files
```

## Module Usage

### Data Loader

```python
from data_loader import LayoutLMDataLoader

loader = LayoutLMDataLoader(data_dir="data")
loader.download_dataset()  # Creates sample data if Kaggle not available
documents = loader.load_documents()
df = loader.load_as_dataframe()
```

### NER Extractor

```python
from ner_extractor import DocumentNER

ner = DocumentNER()
entities = ner.extract_entities(text)
results = ner.extract_from_documents(documents)
summary = ner.get_entity_summary(results)
```

### Text Analyzer

```python
from text_analyzer import TextAnalyzer

analyzer = TextAnalyzer(language='en')
report = analyzer.generate_analysis_report(texts)
vocabulary = analyzer.analyze_vocabulary(texts)
term_freq = analyzer.analyze_term_frequency(texts)
```

### Report Generator

```python
from report_generator import ReportGenerator

generator = ReportGenerator(language='en')
generator.generate_html_report(
    ner_results=ner_results,
    ner_summary=ner_summary,
    text_analysis=text_analysis,
    output_file='report.html'
)
```

## Data Format

The data loader expects JSON files in the `data/raw/` directory with the following structure:

```json
{
  "id": "doc_001",
  "text": "INVOICE #INV-2024-001\nDate: March 15, 2024\n..."
}
```

If no data is found, the script will automatically create sample documents for demonstration.

## Report Features

The generated HTML report includes:

1. **Summary Statistics**: Overview of documents, entities, and vocabulary
2. **Entity Extraction Results**: Visual charts and tables of extracted entities
3. **Monetary Analysis**: Statistics on monetary amounts found
4. **Date Analysis**: Date range and span information
5. **Term Frequency**: Most frequent terms with visualizations
6. **N-gram Analysis**: Top bigrams and trigrams
7. **Topic Modeling**: Discovered topics and associated words

## Multilingual Support

The report generator supports three languages:

- **English (en)**: Default language
- **Spanish (es)**: Español
- **German (de)**: Deutsch

Switch languages using the `--language` flag or the language selector in the generated report.

## Troubleshooting

### spaCy Model Not Found

If you see an error about missing spaCy model:

```bash
python -m spacy download en_core_web_sm
```

### NLTK Data Missing

The script will automatically download required NLTK data on first run. If you encounter issues:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Kaggle API Issues

If Kaggle download fails, the script will automatically create sample data. To use Kaggle:

1. Ensure `kaggle.json` is in `~/.kaggle/`
2. Check API credentials are correct
3. Verify dataset identifier is correct

### Import Errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Example Output

After running the pipeline, you'll get:

- Console output with progress and summary
- HTML report file with visualizations
- Entity extraction results
- Text analysis statistics

Open the generated HTML file in any web browser to view the full report.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

- LayoutLM dataset from Microsoft
- spaCy for NLP capabilities
- NLTK for text processing
- scikit-learn for machine learning utilities

## Future Enhancements

Potential improvements:

- Support for more document types
- Additional entity types
- More sophisticated topic modeling
- Interactive visualizations
- Export to PDF format
- Additional language support

