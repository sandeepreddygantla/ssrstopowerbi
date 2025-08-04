# 🚀 Enterprise RDL Migration Platform

**AI-Powered SSRS Report Migration with Web Interface**

Transform your SQL Server Reporting Services (SSRS) RDL files into Power BI dashboards with intelligent automation, bulk processing capabilities, and a modern web interface for handling 10,000+ files efficiently.

## 📋 Table of Contents

- [Overview](#overview)
- [🌐 Web Application](#web-application)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Generated Output](#generated-output)
- [AI vs Manual Conversion](#ai-vs-manual-conversion)
- [Supported Data Sources](#supported-data-sources)
- [Migration Process](#migration-process)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Version History](#version-history)

## 📊 Overview

The RDL to Power BI Migration Tool automates the complex process of migrating SSRS reports to Power BI by:

- **Parsing RDL files** to extract data sources, datasets, and report elements
- **Converting SQL queries** to Power Query M language
- **Generating DAX measures** for business intelligence and analytics
- **Creating visual configurations** for charts and tables
- **Providing step-by-step migration guides** for non-technical users
- **Generating validation queries** to ensure data accuracy

### 🎯 Automation Level: **70-85%**

- ✅ **90%** - Data source and query conversion
- ✅ **85%** - DAX measures generation  
- ✅ **80%** - Visual configuration mapping
- ✅ **95%** - Migration documentation
- ⚠️ **Manual** - Final visual layout and formatting

## 🌐 Web Application

### **NEW: Enterprise Web Interface**
The tool now includes a modern web application for handling large-scale migrations:

#### **🚀 Key Web Features:**
- **Bulk File Upload**: Drag & drop up to 10,000 RDL files at once
- **Real-time Progress**: Live migration tracking with WebSocket updates  
- **AI-Powered Analysis**: Similarity detection and consolidation recommendations
- **Smart Consolidation**: Merge similar reports to reduce Power BI dashboard count
- **Interactive Dashboard**: Visual progress tracking and job management
- **Organized Results**: Download structured migration outputs as ZIP files

#### **🖥️ Quick Start:**
```bash
# Setup and install dependencies
python setup.py

# Start the web application
python start_web_app.py

# Access at: http://localhost:5000
```

#### **💼 Enterprise Features:**
- Handle 10,000+ RDL files efficiently
- Intelligent similarity scoring and grouping
- Background processing with job queues
- Real-time WebSocket progress updates
- Organized migration results with ZIP downloads
- Modern responsive UI with Tailwind CSS

## ✨ Features

### 🤖 **AI-Powered Intelligence**
- **GPT-4 Integration** for context-aware conversions
- **Business Intelligence Focus** with KPIs and analytics measures
- **Performance Optimization** following Power BI best practices
- **Error Handling** and data validation built-in

### 🔄 **Comprehensive Conversion**
- **Multi-Data Source Support** (SQL Server, Oracle, ODBC, OLE DB)
- **Complex Query Handling** (JOINs, subqueries, parameters)
- **Time Intelligence** (YTD, MTD, growth rates)
- **Chart Type Mapping** (Column, Bar, Line, Pie charts)

### 📁 **Organized Output**
- **Structured Directory Layout** for enterprise use
- **Ready-to-Use Files** (.m, .dax, .json formats)
- **Comprehensive Documentation** with migration guides
- **Validation Tools** for quality assurance

### 👥 **User-Friendly**
- **Non-Technical Friendly** step-by-step instructions
- **Copy-Paste Ready** code snippets
- **Troubleshooting Guides** for common issues
- **Multiple Tool Versions** (AI-only vs Hybrid)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RDL Parser    │───▶│  AI Converter    │───▶│  File Generator │
│                 │    │                  │    │                 │
│ • Data Sources  │    │ • Power Query M  │    │ • .m files      │
│ • Datasets      │    │ • DAX Measures   │    │ • .dax files    │
│ • Report Items  │    │ • Visual Configs │    │ • .json files   │
│ • Complexity    │    │ • Optimization   │    │ • .md guides    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Namespace     │    │   OpenAI GPT-4   │    │ migration_output│
│   Detection     │    │   Integration    │    │   Directory     │
│                 │    │                  │    │                 │
│ • 2008/2010/2016│    │ • Context Aware  │    │ • power_query/  │
│ • Auto-detect   │    │ • Business Focus │    │ • dax_measures/ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🧩 Core Components

#### **1. RDLParser Class**
- Handles multiple RDL namespace versions (2008, 2010, 2016)
- Extracts comprehensive report metadata
- Calculates complexity scores for intelligent processing

#### **2. PowerBIConverter Class**
- AI-powered or manual conversion methods
- Context-aware query optimization
- Business intelligence focused DAX generation

#### **3. MigrationGuideGenerator Class**
- Creates user-friendly documentation
- Generates step-by-step instructions
- Provides troubleshooting guidance

#### **4. File Management System**
- Organized directory structure
- Metadata preservation
- Cross-reference tracking

## 🛠️ Installation

### Prerequisites

```bash
# Required Python packages
pip install xml.etree.ElementTree
pip install json
pip install pathlib
pip install datetime
pip install dataclasses
pip install typing

# Optional: For AI features
pip install openai
pip install python-dotenv

# Optional: For database connectivity (if using demo features)
pip install pyodbc
pip install pandas
```

### Environment Setup

1. **Clone or Download** the tool files
2. **Set OpenAI API Key** (for AI features):

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-api-key-here"

# Option 2: .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

3. **Verify Installation**:

```bash
python rdl_migration_tool.py --help
```

## 🚀 Usage

### Basic Usage

```bash
# Process your RDL file with hybrid AI/manual conversion
python rdl_migration_tool.py path/to/your/report.rdl

# AI-only conversion (requires OpenAI API key)
python rdl_migration_tool_ai_only.py path/to/your/report.rdl

# Force AI conversion even without API key (will show warning)
python rdl_migration_tool.py path/to/your/report.rdl --ai
```

### Example Commands

```bash
# Basic migration
python rdl_migration_tool.py SalesReport.rdl

# With AI enhancement
python rdl_migration_tool_ai_only.py ComplexFinancialReport.rdl

# Process multiple reports (bash)
for file in *.rdl; do
    python rdl_migration_tool.py "$file"
done
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `rdl_file` | Path to RDL file (required) | `SalesReport.rdl` |
| `--ai` | Force AI conversion | `--ai` |
| `--help` | Show help message | `--help` |

## 📁 Generated Output

### Directory Structure

```
migration_output/
├── power_query/              # Power Query M files
│   ├── SalesDataset.m        # Data import code
│   ├── CategorySummary.m     # Aggregated data code
│   └── CustomerAnalysis.m    # Customer data code
│
├── dax_measures/             # DAX measure files
│   ├── SalesDetailTable.dax  # Sales metrics & KPIs
│   ├── CustomerTable.dax     # Customer analytics
│   └── ProductTable.dax      # Product performance
│
├── visual_configs/           # Visual configuration files
│   ├── SalesChart.json       # Chart setup instructions
│   ├── RevenueChart.json     # Revenue visualization
│   └── TrendChart.json       # Trend analysis config
│
├── guides/                   # Migration documentation
│   ├── migration_guide.md    # Comprehensive guide
│   └── troubleshooting.md    # Common issues & solutions
│
├── validation/               # Quality assurance
│   ├── validation_queries.sql # Data verification queries
│   └── test_scenarios.md     # Testing instructions
│
└── documentation/            # Project metadata
    ├── migration_summary.json # Complete analysis results
    ├── complexity_analysis.json # Complexity scoring
    └── data_mapping.json     # Field mappings
```

### File Types Explained

#### **Power Query (.m files)**
- **Purpose**: Data import and transformation
- **Content**: Connection strings, SQL queries, data type definitions
- **Usage**: Copy-paste into Power BI Advanced Editor

```m
// Example SalesDataset.m
let
    Source = Sql.Database("ServerName", "DatabaseName"),
    CustomSQL = Value.NativeQuery(Source, "SELECT * FROM Sales"),
    TypedData = Table.TransformColumnTypes(CustomSQL, {...})
in
    TypedData
```

#### **DAX Measures (.dax files)**
- **Purpose**: Business calculations and KPIs
- **Content**: Measures, calculated columns, time intelligence
- **Usage**: Copy individual measures into Power BI

```dax
// Example from SalesDetailTable.dax
Total Sales = SUM(Sales[Amount])
Sales Growth = DIVIDE([Total Sales] - [Sales Previous Month], [Sales Previous Month])
Top Customer = CALCULATE(VALUES(Sales[Customer]), TOPN(1, Sales, [Total Sales]))
```

#### **Visual Configurations (.json files)**
- **Purpose**: Chart and table setup instructions
- **Content**: Field mappings, formatting options, visual types
- **Usage**: Reference for manual visual creation

```json
{
  "visualType": "clusteredColumnChart",
  "fieldMappings": {
    "axis": "Category",
    "values": ["Total Sales", "Quantity"],
    "legend": "Region"
  },
  "powerBIInstructions": [...]
}
```

## 🤖 AI vs Manual Conversion

### Two Tool Versions Available

| Feature | AI-Only Version | Hybrid Version |
|---------|----------------|----------------|
| **OpenAI Requirement** | ✅ Required | ⚠️ Optional |
| **Conversion Quality** | 🌟 Excellent | 📊 Good |
| **Reliability** | 🌐 Internet dependent | 🔒 Always works |
| **Cost** | 💰 API costs | 🆓 Free |
| **Enterprise Ready** | ⚠️ Limited | ✅ Yes |
| **Code Complexity** | 🎯 Simple | 📚 Comprehensive |

### When to Use Which Version

#### **Use AI-Only Version When:**
- You have reliable OpenAI API access
- Quality is more important than reliability
- You want the cleanest, most intelligent conversions
- You're working on complex reports with intricate business logic

#### **Use Hybrid Version When:**
- You need guaranteed tool functionality
- Working in corporate/restricted environments
- API costs are a concern
- You want maximum compatibility

### AI Enhancement Features

#### **Context-Aware Conversion**
- Understands data source types (SQL Server, Oracle, etc.)
- Recognizes business domains (Sales, Finance, Customer)
- Adapts conversion patterns based on complexity

#### **Business Intelligence Focus**
- Generates KPIs and performance metrics
- Creates time intelligence measures
- Includes trend analysis and growth calculations
- Provides data quality validation measures

#### **Performance Optimization**
- Follows Power BI best practices
- Optimizes DAX for performance
- Implements proper error handling
- Suggests query folding opportunities

## 🔌 Supported Data Sources

### Primary Support
- **SQL Server** - Full support with Windows/SQL authentication
- **Oracle** - Connection strings and query optimization
- **ODBC** - Generic ODBC data source connections
- **OLE DB** - Legacy data source support

### Connection String Parsing
- **Automatic detection** of server, database, authentication
- **Multi-format support** for various connection string patterns
- **Security handling** for integrated vs. database authentication

### Data Source Mapping

| SSRS Data Source | Power BI Connector | Conversion Method |
|------------------|-------------------|-------------------|
| SQL Server | Sql.Database() | Direct mapping |
| Oracle | Oracle.Database() | Query adaptation |
| ODBC | Odbc.DataSource() | Generic connection |
| OLE DB | OleDb.DataSource() | Legacy support |
| Web Service | Web.Contents() | Manual configuration |

## 📋 Migration Process

### Phase 1: Analysis (Automated)
1. **RDL Parsing** - Extract all report components
2. **Complexity Analysis** - Score migration difficulty
3. **Data Source Detection** - Identify connection types
4. **Dependency Mapping** - Understand relationships

### Phase 2: Conversion (AI/Manual)
1. **Query Conversion** - SQL to Power Query M
2. **Measure Generation** - SSRS expressions to DAX
3. **Visual Mapping** - Charts to Power BI visuals
4. **Optimization** - Performance and best practices

### Phase 3: Documentation (Automated)
1. **Guide Generation** - Step-by-step instructions
2. **Validation Queries** - Data accuracy verification
3. **Troubleshooting** - Common issues and solutions
4. **Metadata Preservation** - Complete audit trail

### Phase 4: Implementation (Manual)
1. **Power BI Setup** - Connect to data sources
2. **Code Implementation** - Import generated files
3. **Visual Creation** - Build dashboards
4. **Testing & Validation** - Verify accuracy

## 🔧 Troubleshooting

### Common Issues

#### **RDL Parsing Errors**
```
Error: namespace not found
Solution: Tool automatically detects RDL versions (2008/2010/2016)
Check: Ensure RDL file is valid XML
```

#### **AI Conversion Failures**
```
Error: OpenAI API error
Solution: Check API key configuration and network connectivity
Fallback: Use hybrid version for manual conversion
```

#### **Data Source Connection Issues**
```
Error: Connection string parsing failed
Solution: Manually verify server and database names in generated files
Check: Authentication method compatibility
```

#### **Performance Issues**
```
Error: Large dataset processing slow
Solution: Consider DirectQuery instead of Import mode
Optimization: Use data source-specific optimizations
```

### Debug Mode

Add debug output to troubleshoot issues:

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python rdl_migration_tool.py report.rdl --verbose
```

### Support Resources

- **Generated Documentation** - Check migration_output/guides/
- **Validation Queries** - Use migration_output/validation/
- **Microsoft Documentation** - Power BI and DAX references
- **Community Forums** - Power BI community support

## 📖 API Reference

### Core Classes

#### **RDLParser**
```python
class RDLParser:
    def __init__(self, rdl_file_path: str)
    def extract_data_sources(self) -> List[Dict]
    def extract_datasets(self) -> List[Dict]
    def extract_report_items(self) -> List[ReportElement]
    def _detect_namespace(self) -> dict
    def _calculate_dataset_complexity(self, dataset_info: dict) -> int
```

#### **PowerBIConverter**
```python
class PowerBIConverter:
    def __init__(self, openai_client=None, version="1.0+")
    def convert_sql_to_powerquery(self, sql_query: str, dataset_name: str, data_source_info: dict) -> str
    def convert_table_to_dax(self, table_element: ReportElement) -> str
    def convert_chart_to_powerbi_visual(self, chart_element: ReportElement) -> Dict
```

#### **ReportElement**
```python
@dataclass
class ReportElement:
    name: str
    type: str  # 'Table', 'Chart', 'TextBox'
    properties: Dict[str, Any]
    sql_query: str = ""
    parameters: List[Dict] = None
```

### Configuration Options

#### **Environment Variables**
- `OPENAI_API_KEY` - OpenAI API key for AI features
- `OPENAI_MODEL` - AI model selection (default: gpt-4)
- `OUTPUT_DIR` - Custom output directory (default: ./migration_output)

#### **Tool Parameters**
- `rdl_file_path` - Path to input RDL file
- `ai_enabled` - Enable/disable AI conversion
- `output_directory` - Custom output location
- `complexity_threshold` - Migration complexity scoring

### Return Values

#### **Migration Summary**
```python
{
    "report_name": str,
    "complexity_score": int,
    "data_sources": List[Dict],
    "datasets": List[Dict],
    "report_items": List[Dict],
    "generated_files": Dict[str, List[str]],
    "migration_timestamp": str,
    "ai_powered": bool
}
```

#### **Conversion Results**
```python
{
    "power_query": Dict[str, str],      # {dataset_name: m_code}
    "dax_measures": Dict[str, str],     # {table_name: dax_code}
    "visual_configs": Dict[str, Dict],  # {chart_name: config}
    "validation_queries": str,
    "migration_guide": str
}
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
3. **Set up development environment**:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black rdl_migration_tool.py

# Type checking
mypy rdl_migration_tool.py
```

### Code Structure

```
├── rdl_migration_tool.py          # Main hybrid tool
├── rdl_migration_tool_ai_only.py  # AI-only version
├── tests/                         # Unit tests
│   ├── test_rdl_parser.py
│   ├── test_converter.py
│   └── sample_rdl_files/
├── examples/                      # Example RDL files
├── docs/                         # Additional documentation
└── requirements.txt              # Dependencies
```

### Adding New Features

#### **New Data Source Support**
1. Update `_parse_connection_string()` method
2. Add connector mapping in `convert_sql_to_powerquery()`
3. Test with sample connection strings
4. Update documentation

#### **Enhanced DAX Patterns**
1. Extend `_generate_domain_specific_dax()` method
2. Add business domain detection
3. Include new measure patterns
4. Update AI prompts for context

#### **Visual Type Mapping**
1. Update `convert_chart_to_powerbi_visual()` method
2. Add new chart type mappings
3. Include formatting recommendations
4. Test with various chart configurations

### Testing

#### **Unit Tests**
```bash
# Run all tests
python -m pytest

# Test specific component
python -m pytest tests/test_rdl_parser.py

# Test with sample files
python -m pytest tests/test_integration.py
```

#### **Integration Tests**
```bash
# Test with real RDL files
python rdl_migration_tool.py tests/sample_rdl_files/sales_report.rdl

# Validate output structure
python tests/validate_output.py migration_output/
```

## 📋 Version History

### Version 2.0.0 - AI-Powered Release
**Release Date**: 2025-01-03

#### ✨ **New Features**
- **GPT-4 Integration** for intelligent conversions
- **Context-Aware Processing** based on data source types
- **Business Intelligence Focus** with KPIs and analytics
- **Dual Tool Architecture** (AI-only vs Hybrid versions)
- **Enhanced Documentation** with step-by-step guides

#### 🔧 **Improvements**
- **Universal Data Source Support** (SQL Server, Oracle, ODBC, OLE DB)
- **Advanced RDL Parsing** with namespace auto-detection
- **Organized Output Structure** for enterprise use
- **Comprehensive Validation** with SQL verification queries
- **Performance Optimization** following Power BI best practices

#### 🐛 **Bug Fixes**
- Fixed namespace detection for RDL 2008/2010/2016 versions
- Resolved connection string parsing for various formats
- Improved error handling and user feedback
- Enhanced file path handling for cross-platform compatibility

### Version 1.0.0 - Initial Release
**Release Date**: 2024-12-15

#### ✨ **Features**
- Basic RDL file parsing
- SQL to Power Query conversion
- Simple DAX measure generation
- Manual conversion methods
- Basic output file generation

---

## 📞 Support & Contact

### Getting Help

1. **Check Documentation** - Review this README and generated guides
2. **Run Validation** - Use generated validation queries
3. **Review Examples** - Check sample outputs and configurations
4. **Community Support** - Power BI community forums

### Reporting Issues

When reporting issues, please include:
- **Tool Version** - Which version you're using
- **RDL File Info** - RDL version and complexity
- **Error Messages** - Complete error output
- **Environment** - OS, Python version, dependencies
- **Sample Data** - Anonymized RDL snippet if possible

### Feature Requests

We welcome suggestions for:
- **New Data Source Support**
- **Enhanced AI Prompting**
- **Additional DAX Patterns**
- **Visual Type Mappings**
- **Performance Optimizations**

---

## 📜 License

This tool is provided as-is for educational and professional use. Please ensure compliance with your organization's policies regarding:
- **AI Service Usage** (OpenAI API)
- **Data Processing** (RDL file contents)
- **Tool Deployment** (internal vs external use)

---

**🚀 Transform Your SSRS Reports to Power BI with Intelligence!**

*Generated and maintained by the RDL to Power BI Migration Tool team*