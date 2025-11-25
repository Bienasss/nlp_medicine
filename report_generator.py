"""
Report generation module with HTML output and multilingual support.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class MultilingualTranslator:
    """Simple translation dictionary for report labels."""
    
    TRANSLATIONS = {
        'en': {
            'title': 'Document Understanding Analysis Report',
            'generated': 'Generated on',
            'summary': 'Summary',
            'documents_analyzed': 'Documents Analyzed',
            'entities_extracted': 'Entities Extracted',
            'vocabulary_stats': 'Vocabulary Statistics',
            'total_tokens': 'Total Tokens',
            'unique_tokens': 'Unique Tokens',
            'avg_tokens': 'Average Tokens per Document',
            'entity_extraction': 'Entity Extraction Results',
            'entity_type': 'Entity Type',
            'count': 'Count',
            'monetary_analysis': 'Monetary Amount Analysis',
            'total_amount': 'Total Amount',
            'average_amount': 'Average Amount',
            'min_amount': 'Minimum Amount',
            'max_amount': 'Maximum Amount',
            'date_analysis': 'Date Analysis',
            'earliest_date': 'Earliest Date',
            'latest_date': 'Latest Date',
            'date_span': 'Date Span (days)',
            'term_frequency': 'Term Frequency Analysis',
            'term': 'Term',
            'frequency': 'Frequency',
            'document_frequency': 'Document Frequency',
            'top_terms': 'Top Terms',
            'cooccurrence': 'Co-occurrence Patterns',
            'bigrams': 'Top Bigrams',
            'trigrams': 'Top Trigrams',
            'topic_modeling': 'Topic Modeling',
            'topic': 'Topic',
            'top_words': 'Top Words',
            'no_data': 'No data available',
            'invoice_number': 'Invoice Number',
            'receipt_number': 'Receipt Number',
            'transaction_id': 'Transaction ID',
            'po_number': 'PO Number',
            'account_number': 'Account Number',
            'monetary_amount': 'Monetary Amount',
            'date': 'Date',
            'email': 'Email',
            'phone': 'Phone',
            'address': 'Address',
        },
        'es': {
            'title': 'Informe de Análisis de Comprensión de Documentos',
            'generated': 'Generado el',
            'summary': 'Resumen',
            'documents_analyzed': 'Documentos Analizados',
            'entities_extracted': 'Entidades Extraídas',
            'vocabulary_stats': 'Estadísticas de Vocabulario',
            'total_tokens': 'Total de Tokens',
            'unique_tokens': 'Tokens Únicos',
            'avg_tokens': 'Promedio de Tokens por Documento',
            'entity_extraction': 'Resultados de Extracción de Entidades',
            'entity_type': 'Tipo de Entidad',
            'count': 'Cantidad',
            'monetary_analysis': 'Análisis de Montos Monetarios',
            'total_amount': 'Monto Total',
            'average_amount': 'Monto Promedio',
            'min_amount': 'Monto Mínimo',
            'max_amount': 'Monto Máximo',
            'date_analysis': 'Análisis de Fechas',
            'earliest_date': 'Fecha Más Antigua',
            'latest_date': 'Fecha Más Reciente',
            'date_span': 'Rango de Fechas (días)',
            'term_frequency': 'Análisis de Frecuencia de Términos',
            'term': 'Término',
            'frequency': 'Frecuencia',
            'document_frequency': 'Frecuencia en Documentos',
            'top_terms': 'Términos Principales',
            'cooccurrence': 'Patrones de Co-ocurrencia',
            'bigrams': 'Bigramas Principales',
            'trigrams': 'Trigramas Principales',
            'topic_modeling': 'Modelado de Temas',
            'topic': 'Tema',
            'top_words': 'Palabras Principales',
            'no_data': 'No hay datos disponibles',
            'invoice_number': 'Número de Factura',
            'receipt_number': 'Número de Recibo',
            'transaction_id': 'ID de Transacción',
            'po_number': 'Número de PO',
            'account_number': 'Número de Cuenta',
            'monetary_amount': 'Monto Monetario',
            'date': 'Fecha',
            'email': 'Correo Electrónico',
            'phone': 'Teléfono',
            'address': 'Dirección',
        },
        'de': {
            'title': 'Dokumentverständnis-Analysebericht',
            'generated': 'Erstellt am',
            'summary': 'Zusammenfassung',
            'documents_analyzed': 'Analysierte Dokumente',
            'entities_extracted': 'Extrahierte Entitäten',
            'vocabulary_stats': 'Wortschatzstatistiken',
            'total_tokens': 'Gesamte Tokens',
            'unique_tokens': 'Eindeutige Tokens',
            'avg_tokens': 'Durchschnittliche Tokens pro Dokument',
            'entity_extraction': 'Entitätsextraktionsergebnisse',
            'entity_type': 'Entitätstyp',
            'count': 'Anzahl',
            'monetary_analysis': 'Geldbetragsanalyse',
            'total_amount': 'Gesamtbetrag',
            'average_amount': 'Durchschnittsbetrag',
            'min_amount': 'Mindestbetrag',
            'max_amount': 'Höchstbetrag',
            'date_analysis': 'Datenanalyse',
            'earliest_date': 'Frühestes Datum',
            'latest_date': 'Spätestes Datum',
            'date_span': 'Datumsbereich (Tage)',
            'term_frequency': 'Begriffshäufigkeitsanalyse',
            'term': 'Begriff',
            'frequency': 'Häufigkeit',
            'document_frequency': 'Dokumenthäufigkeit',
            'top_terms': 'Top-Begriffe',
            'cooccurrence': 'Kookkurrenzmuster',
            'bigrams': 'Top-Bigramme',
            'trigrams': 'Top-Trigramme',
            'topic_modeling': 'Themenmodellierung',
            'topic': 'Thema',
            'top_words': 'Top-Wörter',
            'no_data': 'Keine Daten verfügbar',
            'invoice_number': 'Rechnungsnummer',
            'receipt_number': 'Quittungsnummer',
            'transaction_id': 'Transaktions-ID',
            'po_number': 'PO-Nummer',
            'account_number': 'Kontonummer',
            'monetary_amount': 'Geldbetrag',
            'date': 'Datum',
            'email': 'E-Mail',
            'phone': 'Telefon',
            'address': 'Adresse',
        }
    }
    
    @classmethod
    def translate(cls, key: str, lang: str = 'en') -> str:
        """Get translation for a key."""
        return cls.TRANSLATIONS.get(lang, cls.TRANSLATIONS['en']).get(key, key)
    
    @classmethod
    def get_entity_label(cls, entity_type: str, lang: str = 'en') -> str:
        """Get translated label for entity type."""
        return cls.translate(entity_type, lang)


class ReportGenerator:
    """Generates HTML reports with visualizations."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize report generator.
        
        Args:
            language: Language code ('en', 'es', 'de')
        """
        self.language = language
        self.translator = MultilingualTranslator()
    
    def _create_plot_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        return img_str
    
    def _create_entity_count_chart(self, entity_counts: Dict[str, int]) -> str:
        """Create bar chart for entity counts."""
        if not entity_counts:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        entity_types = list(entity_counts.keys())
        counts = list(entity_counts.values())
        
        bars = ax.barh(entity_types, counts, color='steelblue')
        ax.set_xlabel(self.translator.translate('count', self.language))
        ax.set_ylabel(self.translator.translate('entity_type', self.language))
        ax.set_title(self.translator.translate('entity_extraction', self.language))
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center')
        
        plt.tight_layout()
        return self._create_plot_base64(fig)
    
    def _create_term_frequency_chart(self, term_freq_data: List[Dict]) -> str:
        """Create bar chart for term frequency."""
        if not term_freq_data:
            return ""
        
        df = pd.DataFrame(term_freq_data)
        top_10 = df.head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_10['term'], top_10['frequency'], color='coral')
        ax.set_xlabel(self.translator.translate('frequency', self.language))
        ax.set_ylabel(self.translator.translate('term', self.language))
        ax.set_title(self.translator.translate('top_terms', self.language))
        
        plt.tight_layout()
        return self._create_plot_base64(fig)
    
    def _create_monetary_chart(self, monetary_data: Dict[str, float]) -> str:
        """Create visualization for monetary analysis."""
        if not monetary_data or 'sum' not in monetary_data:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        metrics = ['sum', 'average', 'min', 'max']
        labels = [
            self.translator.translate('total_amount', self.language),
            self.translator.translate('average_amount', self.language),
            self.translator.translate('min_amount', self.language),
            self.translator.translate('max_amount', self.language)
        ]
        values = [monetary_data.get(m, 0) for m in metrics]
        
        bars = ax.bar(labels, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        ax.set_ylabel('USD ($)')
        ax.set_title(self.translator.translate('monetary_analysis', self.language))
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val:,.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return self._create_plot_base64(fig)
    
    def generate_html_report(self, 
                            ner_results: List[Dict[str, Any]],
                            ner_summary: Dict[str, Any],
                            text_analysis: Dict[str, Any],
                            output_file: str = 'report.html') -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            ner_results: NER extraction results
            ner_summary: NER summary statistics
            text_analysis: Text analysis results
            output_file: Output file path
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating HTML report in {self.language}...")
        
        # Create visualizations
        entity_chart = self._create_entity_count_chart(ner_summary.get('entity_type_counts', {}))
        term_freq_chart = self._create_term_frequency_chart(text_analysis.get('term_frequency', []))
        monetary_chart = self._create_monetary_chart(ner_summary.get('monetary_totals', {}))
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="{self.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.translator.translate('title', self.language)}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin-bottom: 10px;
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .entity-list {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .entity-item {{
            margin: 5px 0;
            padding: 5px;
            background: white;
            border-left: 3px solid #3498db;
            padding-left: 10px;
        }}
        .language-selector {{
            text-align: right;
            margin-bottom: 20px;
        }}
        .language-selector a {{
            margin: 0 5px;
            padding: 5px 10px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 3px;
        }}
        .language-selector a:hover {{
            background: #2980b9;
        }}
        .metadata {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .topic-box {{
            background: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #e74c3c;
        }}
        .topic-box h4 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .topic-words {{
            color: #34495e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="language-selector">
            <a href="?lang=en">English</a>
            <a href="?lang=es">Español</a>
            <a href="?lang=de">Deutsch</a>
        </div>
        
        <h1>{self.translator.translate('title', self.language)}</h1>
        <div class="metadata">
            {self.translator.translate('generated', self.language)}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <h2>{self.translator.translate('summary', self.language)}</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{self.translator.translate('documents_analyzed', self.language)}</h3>
                <div class="value">{ner_summary.get('total_documents', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>{self.translator.translate('entities_extracted', self.language)}</h3>
                <div class="value">{ner_summary.get('total_entities', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>{self.translator.translate('total_tokens', self.language)}</h3>
                <div class="value">{text_analysis.get('vocabulary', {}).get('total_tokens', 0):,}</div>
            </div>
            <div class="summary-card">
                <h3>{self.translator.translate('unique_tokens', self.language)}</h3>
                <div class="value">{text_analysis.get('vocabulary', {}).get('unique_tokens', 0):,}</div>
            </div>
        </div>
        
        <h2>{self.translator.translate('entity_extraction', self.language)}</h2>
        {f'<div class="chart-container"><img src="data:image/png;base64,{entity_chart}" alt="Entity Count Chart"></div>' if entity_chart else '<p>' + self.translator.translate('no_data', self.language) + '</p>'}
        
        <table>
            <thead>
                <tr>
                    <th>{self.translator.translate('entity_type', self.language)}</th>
                    <th>{self.translator.translate('count', self.language)}</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add entity counts table
        for entity_type, count in ner_summary.get('entity_type_counts', {}).items():
            label = self.translator.get_entity_label(entity_type, self.language)
            html_content += f"""
                <tr>
                    <td>{label}</td>
                    <td>{count}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
"""
        
        # Monetary analysis
        if ner_summary.get('monetary_totals'):
            html_content += f"""
        <h2>{self.translator.translate('monetary_analysis', self.language)}</h2>
        {f'<div class="chart-container"><img src="data:image/png;base64,{monetary_chart}" alt="Monetary Analysis Chart"></div>' if monetary_chart else ''}
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
"""
            monetary = ner_summary['monetary_totals']
            html_content += f"""
                <tr>
                    <td>{self.translator.translate('total_amount', self.language)}</td>
                    <td>${monetary.get('sum', 0):,.2f}</td>
                </tr>
                <tr>
                    <td>{self.translator.translate('average_amount', self.language)}</td>
                    <td>${monetary.get('average', 0):,.2f}</td>
                </tr>
                <tr>
                    <td>{self.translator.translate('min_amount', self.language)}</td>
                    <td>${monetary.get('min', 0):,.2f}</td>
                </tr>
                <tr>
                    <td>{self.translator.translate('max_amount', self.language)}</td>
                    <td>${monetary.get('max', 0):,.2f}</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
"""
        
        # Date analysis
        if ner_summary.get('date_range'):
            html_content += f"""
        <h2>{self.translator.translate('date_analysis', self.language)}</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
"""
            date_range = ner_summary['date_range']
            html_content += f"""
                <tr>
                    <td>{self.translator.translate('earliest_date', self.language)}</td>
                    <td>{date_range.get('earliest', 'N/A')}</td>
                </tr>
                <tr>
                    <td>{self.translator.translate('latest_date', self.language)}</td>
                    <td>{date_range.get('latest', 'N/A')}</td>
                </tr>
                <tr>
                    <td>{self.translator.translate('date_span', self.language)}</td>
                    <td>{date_range.get('span_days', 0)}</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
"""
        
        # Term frequency
        if text_analysis.get('term_frequency'):
            html_content += f"""
        <h2>{self.translator.translate('term_frequency', self.language)}</h2>
        {f'<div class="chart-container"><img src="data:image/png;base64,{term_freq_chart}" alt="Term Frequency Chart"></div>' if term_freq_chart else ''}
        <table>
            <thead>
                <tr>
                    <th>{self.translator.translate('term', self.language)}</th>
                    <th>{self.translator.translate('frequency', self.language)}</th>
                    <th>{self.translator.translate('document_frequency', self.language)}</th>
                </tr>
            </thead>
            <tbody>
"""
            for term_data in text_analysis['term_frequency'][:20]:
                html_content += f"""
                <tr>
                    <td>{term_data.get('term', '')}</td>
                    <td>{term_data.get('frequency', 0)}</td>
                    <td>{term_data.get('document_frequency', 0)}</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
"""
        
        # Bigrams and Trigrams
        if text_analysis.get('bigrams'):
            html_content += f"""
        <h2>{self.translator.translate('bigrams', self.language)}</h2>
        <div class="entity-list">
"""
            for bigram, count in text_analysis['bigrams'][:15]:
                html_content += f'<div class="entity-item"><strong>{bigram}</strong>: {count}</div>\n'
            html_content += "</div>"
        
        if text_analysis.get('trigrams'):
            html_content += f"""
        <h2>{self.translator.translate('trigrams', self.language)}</h2>
        <div class="entity-list">
"""
            for trigram, count in text_analysis['trigrams'][:15]:
                html_content += f'<div class="entity-item"><strong>{trigram}</strong>: {count}</div>\n'
            html_content += "</div>"
        
        # Topic modeling
        if text_analysis.get('topics'):
            html_content += f"""
        <h2>{self.translator.translate('topic_modeling', self.language)}</h2>
"""
            for topic_name, words in text_analysis['topics'].items():
                html_content += f"""
        <div class="topic-box">
            <h4>{topic_name}</h4>
            <div class="topic-words">{', '.join(words)}</div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {output_file}")
        return output_file


if __name__ == "__main__":
    # Example usage
    generator = ReportGenerator(language='en')
    print("Report generator initialized")

