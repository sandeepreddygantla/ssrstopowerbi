# 🚀 SSRS to Power BI Migration Tool

**AI-Powered Enterprise Migration Platform with Token-Optimized Batch Processing**

Transform your SQL Server Reporting Services (SSRS) RDL files into Power BI dashboards with intelligent automation, bulk processing capabilities, and advanced business logic analysis for handling thousands of files efficiently.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Business Logic Analysis](#business-logic-analysis)
- [Token-Optimized Migration Guides](#token-optimized-migration-guides)
- [Generated Output](#generated-output)
- [Supported Data Sources](#supported-data-sources)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 📊 Overview

The SSRS to Power BI Migration Tool automates the complex process of migrating enterprise SSRS reports to Power BI through:

- **Advanced RDL Parsing** with multi-namespace support (2008, 2010, 2016)
- **AI-Powered Business Logic Analysis** for consolidation recommendations
- **Token-Optimized Batch Processing** to minimize LLM costs
- **Intelligent SQL Query Conversion** to Power Query M language
- **Context-Aware DAX Generation** with business intelligence focus
- **Similarity Analysis Engine** for report consolidation opportunities
- **Enterprise Web Interface** for bulk migration management

### 🎯 Enterprise Automation Level: **75-90%**

- ✅ **95%** - RDL parsing and data extraction
- ✅ **90%** - SQL to Power Query conversion
- ✅ **85%** - Business logic similarity analysis
- ✅ **80%** - DAX measures generation
- ✅ **90%** - Migration documentation
- ⚠️ **Manual** - Final visual layout and formatting

## ✨ Key Features

### 🤖 **AI-Powered Intelligence**
- **GPT-4 Integration** for context-aware conversions
- **Business Logic Similarity Analysis** with 5-metric scoring system
- **Token-Optimized Batch Guides** reducing LLM usage by 60-80%
- **Performance Optimization** following Power BI best practices

### 🏢 **Enterprise Web Interface**
- **Bulk File Upload** with drag & drop for thousands of files
- **Real-time Progress Tracking** with WebSocket updates
- **Advanced Similarity Analysis** with consolidation recommendations
- **Organized Results Management** with structured downloads
- **Professional UI** with responsive design

### 📊 **Business Logic Analysis Engine**
- **5-Metric Analysis System**: Data Source, Filter Logic, Business Purpose, Calculations, Parameters
- **Similarity Scoring** with percentage-based matching (High ≥70%, Medium 40-69%, Low <40%)
- **SQL Query Comparison** for manual validation
- **Consolidation Recommendations** based on functional similarity

### 🔄 **Advanced Migration Capabilities**
- **Multi-Database Support** (SQL Server, Oracle, T-SQL, MySQL, PostgreSQL)
- **Complex Query Handling** with joins, subqueries, and parameters
- **Time Intelligence** with YTD, MTD, and growth calculations
- **Chart Type Mapping** for comprehensive visual conversion

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key (optional but recommended)
export OPENAI_API_KEY="your-api-key-here"
# Or create .env file with: OPENAI_API_KEY=your-api-key-here
```

### Start the Web Application
```bash
# Start the migration platform
python web_app.py

# Access the tool at: http://localhost:5000/migration
```

### Basic Workflow
1. **Upload RDL Files** - Drag and drop your SSRS files
2. **Configure Analysis** - Set similarity thresholds and options
3. **Run Migration** - Let the AI process and convert your reports
4. **Review Results** - Analyze similarity scores and consolidation opportunities
5. **Download Assets** - Get organized Power Query, DAX, and migration guides

## 🏗️ Architecture

### Core System Design
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │───▶│  RDL Parser      │───▶│ Business Logic  │
│   (Flask+SocketIO)│    │  Multi-Namespace │    │ Analyzer        │
│                 │    │                  │    │                 │
│ • File Upload   │    │ • Data Sources   │    │ • 5-Metric      │
│ • Progress      │    │ • Datasets       │    │   Analysis      │
│ • Results UI    │    │ • Report Items   │    │ • Similarity    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Token-Optimized│    │   AI Converter   │    │  File Generator │
│  Guide Generator│    │   (GPT-4)        │    │                 │
│                 │    │                  │    │ • .m files      │
│ • Batch Guides  │    │ • Power Query M  │    │ • .dax files    │
│ • Dynamic       │    │ • DAX Measures   │    │ • .json configs │
│   Content       │    │ • Optimization   │    │ • Migration     │
│ • Context-Aware │    │ • Best Practices │    │   Guides        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

#### **1. Web Application (web_app.py)**
- Flask application with SocketIO for real-time updates
- Background job processing with threading
- Secure file upload handling (500MB max)
- REST API endpoints for analysis and migration

#### **2. RDL Business Analyzer (rdl_business_analyzer.py)**
- Enterprise-grade similarity analysis engine
- 5-metric scoring system for functional comparison
- SQL query extraction and comparison
- Consolidation recommendations

#### **3. RDL Parser (app.py)**
- Multi-namespace RDL support (2008, 2010, 2016)
- Comprehensive data extraction
- Business context analysis
- Token-optimized batch guide generation

#### **4. Data Type System (rdl_types.py)**
- Structured data classes for business logic
- Enum definitions for analysis categories
- Helper utilities for consistent parsing

## 🛠️ Installation

### System Requirements
- Python 3.8+
- 4GB RAM minimum (8GB recommended for large batches)
- OpenAI API access (optional but recommended)

### Dependencies
```bash
# Core requirements
pip install flask flask-socketio
pip install openai python-dotenv
pip install sqlparse scikit-learn
pip install lxml beautifulsoup4

# All dependencies
pip install -r requirements.txt
```

### Environment Configuration
```bash
# Required for AI features
export OPENAI_API_KEY="your-api-key-here"

# Optional configurations
export FLASK_ENV=development
export MAX_CONTENT_LENGTH=500MB
```

## 🚀 Usage

### Web Interface Usage

#### 1. Basic Migration
```bash
# Start the application
python web_app.py

# Navigate to http://localhost:5000/migration
# Upload RDL files using drag & drop
# Configure analysis settings
# Run migration and monitor progress
```

#### 2. Similarity Analysis
- Upload multiple RDL files
- Run similarity analysis to identify consolidation opportunities
- Review detailed comparison reports with SQL query analysis
- Download consolidated migration recommendations

#### 3. Batch Processing
- Process thousands of files efficiently
- Monitor real-time progress with WebSocket updates
- Generate token-optimized migration guides
- Download organized results with structured file layouts

### Command Line Usage
```bash
# Direct file processing
python app.py path/to/report.rdl

# Batch processing
python -c "
from app import RDLParser, PowerBIConverter
parser = RDLParser('report.rdl')
converter = PowerBIConverter()
# Process files...
"
```

## 📊 Business Logic Analysis

### 5-Metric Analysis System

The tool analyzes reports across five critical dimensions:

#### **1. Data Source Similarity (25%)**
- Database server and name matching
- Connection string analysis
- Authentication method comparison

#### **2. Filter Logic Similarity (20%)**
- WHERE clause analysis
- Parameter usage patterns
- Filter condition matching

#### **3. Business Purpose Similarity (20%)**
- Semantic analysis of report context
- Business domain detection
- Functional purpose matching

#### **4. Calculations Similarity (20%)**
- Expression and formula analysis
- Aggregation pattern matching
- Business rule comparison

#### **5. Parameters Similarity (15%)**
- Parameter name and type matching
- Default value comparison
- Usage pattern analysis

### Similarity Scoring
- **High Similarity (≥70%)**: Strong consolidation candidates
- **Medium Similarity (40-69%)**: Consider consolidation with modifications
- **Low Similarity (<40%)**: Keep as separate reports

### Analysis Report Format
The system generates **Analysis Report 154** with:
- Three-tier similarity grouping
- Detailed metric breakdowns
- Side-by-side SQL query comparison
- Consolidation recommendations

## 🎯 Token-Optimized Migration Guides

### Intelligent Batch Processing
Instead of generating individual migration guides for each file (wasteful), the system creates:

- **Single Batch-Level Guide** with populated content for all files
- **Dynamic Context Sections** with file-specific details
- **Shared Migration Steps** to reduce redundancy
- **Smart Asset Linking** with actual file references

### Token Savings
- **60-80% reduction** in LLM token usage
- **47% smaller file sizes** (6.6KB → 3.5KB for 2-file batches)
- **Consistent quality** across all migration documentation
- **Context-aware content** without placeholder variables

### Guide Features
- File context tables with complexity analysis
- Step-by-step migration instructions
- Asset organization with direct file links
- Validation checklists and troubleshooting

## 📁 Generated Output

### Directory Structure
```
results/[batch-id]/
├── [ReportName]/
│   ├── [DatasetName].m          # Power Query files
│   └── [TableName].dax          # DAX measure files
├── guides/
│   └── batch_migration_guide.md # Token-optimized guide
└── analysis/
    ├── similarity_analysis.json # Business logic comparison
    └── consolidation_report.md  # Recommendations
```

### File Types

#### **Power Query (.m files)**
- SQL to Power Query M conversion
- Connection string optimization
- Data type handling
- Error handling and validation

#### **DAX Measures (.dax files)**
- Business KPIs and calculations
- Time intelligence functions
- Performance-optimized measures
- Context-aware formulations

#### **Migration Guides (.md files)**
- Step-by-step implementation instructions
- Visual creation guidance
- Validation procedures
- Troubleshooting assistance

## 🔌 Supported Data Sources

### Primary Support
- **SQL Server** - Full T-SQL support with optimization
- **Oracle** - PL/SQL compatibility with query adaptation
- **MySQL** - Query syntax conversion
- **PostgreSQL** - Advanced feature support
- **ODBC/OLE DB** - Generic connectivity

### Connection Features
- **Automatic Detection** of database types
- **Multi-Format Support** for connection strings
- **Security Handling** for various authentication methods
- **Performance Optimization** for each database type

## 📖 API Reference

### Core Classes

#### **RDLBusinessAnalyzer**
```python
class RDLBusinessAnalyzer:
    def analyze_similarity(self, rdl_files: List[str]) -> Dict
    def extract_business_logic(self, rdl_file: str) -> BusinessLogic
    def calculate_similarity_score(self, logic1: BusinessLogic, logic2: BusinessLogic) -> float
    def generate_consolidation_report(self, analysis_results: Dict) -> str
```

#### **RDLParser**
```python
class RDLParser:
    def __init__(self, rdl_file_path: str)
    def extract_data_sources(self) -> List[Dict]
    def extract_datasets(self) -> List[Dict]
    def extract_report_items(self) -> List[ReportElement]
    def _detect_namespace(self) -> dict
```

#### **PowerBIConverter**
```python
class PowerBIConverter:
    def __init__(self, openai_client=None)
    def convert_sql_to_powerquery(self, sql_query: str, dataset_name: str) -> str
    def convert_table_to_dax(self, table_element: ReportElement) -> str
    def generate_batch_migration_guide(self, batch_info: dict) -> str
```

### Configuration Options
- `OPENAI_API_KEY` - OpenAI API key for AI features
- `MAX_CONTENT_LENGTH` - File upload size limit
- `FLASK_ENV` - Development/production mode
- `SIMILARITY_THRESHOLD` - Analysis sensitivity

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd ssrs-to-powerbi

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run tests
python -m pytest tests/

# Start development server
python web_app.py
```

### Project Structure
```
├── app.py                          # Core RDL parsing and conversion
├── web_app.py                      # Flask web application
├── rdl_business_analyzer.py        # Business logic analysis
├── rdl_parser_helpers.py           # Parsing utilities
├── rdl_types.py                    # Data type definitions
├── llm_config.py                   # AI client configuration
├── templates/                      # HTML templates
│   ├── index-embedded.html
│   ├── migration-embedded.html
│   └── results.html
├── sample_files/                   # Test RDL files
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

### Adding Features
- **New Database Support**: Extend connection string parsing
- **Enhanced Analysis**: Add new similarity metrics
- **UI Improvements**: Enhance web interface
- **Performance**: Optimize batch processing

## 📋 Version History

### Version 3.0.0 - Enterprise Release (Current)
**Release Date**: 2025-08-04

#### ✨ **New Features**
- **Token-Optimized Migration Guides** with 60-80% LLM cost reduction
- **Advanced Business Logic Analysis** with 5-metric scoring
- **Enterprise Web Interface** with bulk processing
- **Real-time Progress Tracking** with WebSocket integration
- **Similarity Analysis Engine** for report consolidation

#### 🔧 **Improvements**
- **Multi-Database Support** beyond SQL Server
- **Enhanced UI/UX** with professional design
- **Batch Processing Optimization** for thousands of files
- **Comprehensive Error Handling** and validation
- **Organized Output Structure** for enterprise deployment

#### 🐛 **Bug Fixes**
- Fixed placeholder variable replacement in migration guides
- Resolved namespace detection issues
- Improved connection string parsing accuracy
- Enhanced cross-platform compatibility

---

## 📞 Support & Contact

### Getting Help
1. **Check Documentation** - Review README and generated guides
2. **Test with Samples** - Use provided sample RDL files
3. **Review Analysis Reports** - Check similarity analysis results
4. **Validate Output** - Use generated validation procedures

### System Requirements
- **Minimum**: 4GB RAM, Python 3.8+
- **Recommended**: 8GB RAM, SSD storage
- **Network**: Internet access for AI features
- **Browser**: Modern browser for web interface

---

**🚀 Transform Your Enterprise SSRS Infrastructure to Power BI with AI Intelligence!**

*Efficient • Intelligent • Enterprise-Ready*