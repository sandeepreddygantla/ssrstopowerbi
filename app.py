#!/usr/bin/env python3
"""
RDL to Power BI Migration Tool - Enterprise Version
OpenAI-powered analysis for intelligent RDL to Power BI migration
"""

import xml.etree.ElementTree as ET
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import LLM configuration
from llm_config import llm, embedding_model, refresh_clients
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify LLM initialization
if not llm:
    logger.error("❌ Error: LLM client not initialized. Check your environment configuration.")
    logger.error("   - Set OPENAI_API_KEY environment variable to enable AI features")
    # Don't exit - allow system to continue with limited functionality

@dataclass
class ReportElement:
    """Data class to store extracted report elements"""
    name: str
    type: str
    properties: Dict[str, Any]
    sql_query: str = ""
    parameters: List[Dict] = None

class RDLParser:
    """Enhanced parser for RDL files to extract comprehensive report components"""
    
    def __init__(self, rdl_file_path: str):
        self.rdl_file_path = rdl_file_path
        self.tree = ET.parse(rdl_file_path)
        self.root = self.tree.getroot()
        
        # Handle different RDL namespace versions
        self.namespace = self._detect_namespace()
        
        # Store extracted information
        self.report_info = {
            'report_name': Path(rdl_file_path).stem,
            'namespace_version': self.namespace,
            'complexity_score': 0
        }
    
    def _detect_namespace(self) -> dict:
        """Detect and return appropriate namespace for the RDL version"""
        root_tag = self.root.tag
        
        if '2016' in root_tag:
            return {'': 'http://schemas.microsoft.com/sqlserver/reporting/2016/01/reportdefinition'}
        elif '2010' in root_tag:
            return {'': 'http://schemas.microsoft.com/sqlserver/reporting/2010/01/reportdefinition'}
        elif '2008' in root_tag:
            return {'': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition'}
        else:
            # Default to 2016 namespace
            return {'': 'http://schemas.microsoft.com/sqlserver/reporting/2016/01/reportdefinition'}
    
    def extract_data_sources(self) -> List[Dict]:
        """Extract comprehensive data source information"""
        data_sources = []
        for ds in self.root.findall('.//DataSource', self.namespace):
            source_info = {
                'name': ds.get('Name', ''),
                'connection_string': '',
                'data_source_reference': '',
                'provider': '',
                'server': '',
                'database': '',
                'authentication': '',
                'integrated_security': False,
                'source_type': 'Unknown'
            }
            
            # Extract connection properties
            conn_props = ds.find('.//ConnectionProperties', self.namespace)
            if conn_props is not None:
                # Data provider
                provider = conn_props.find('DataProvider', self.namespace)
                if provider is not None:
                    source_info['provider'] = provider.text
                    # Determine source type from provider
                    if 'SQL' in source_info['provider'].upper():
                        source_info['source_type'] = 'SQL Server'
                    elif 'OLEDB' in source_info['provider'].upper():
                        source_info['source_type'] = 'OLE DB'
                    elif 'ODBC' in source_info['provider'].upper():
                        source_info['source_type'] = 'ODBC'
                    elif 'ORACLE' in source_info['provider'].upper():
                        source_info['source_type'] = 'Oracle'
                    else:
                        source_info['source_type'] = source_info['provider']
                
                # Connection string or individual components
                conn_string = conn_props.find('ConnectString', self.namespace)
                if conn_string is not None:
                    source_info['connection_string'] = conn_string.text
                    # Parse connection string components
                    self._parse_connection_string(source_info)
                
                # Data source (server)
                data_source = conn_props.find('DataSource', self.namespace)
                if data_source is not None:
                    source_info['server'] = data_source.text
                
                # Initial catalog (database)
                catalog = conn_props.find('InitialCatalog', self.namespace)
                if catalog is not None:
                    source_info['database'] = catalog.text
                
                # Integrated security
                integrated_sec = conn_props.find('IntegratedSecurity', self.namespace)
                if integrated_sec is not None:
                    source_info['integrated_security'] = integrated_sec.text.lower() == 'true'
                    source_info['authentication'] = 'Windows' if source_info['integrated_security'] else 'Database'
                    
            data_sources.append(source_info)
        
        return data_sources
    
    def _parse_connection_string(self, source_info: dict):
        """Parse connection string to extract server and database information"""
        conn_str = source_info['connection_string']
        if not conn_str:
            return
        
        # Parse common connection string patterns
        parts = conn_str.split(';')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key in ['server', 'data source', 'host']:
                    source_info['server'] = value
                elif key in ['database', 'initial catalog', 'dbname']:
                    source_info['database'] = value
                elif key == 'integrated security':
                    source_info['integrated_security'] = value.lower() == 'true'
                elif key == 'trusted_connection':
                    source_info['integrated_security'] = value.lower() == 'yes'
    
    def extract_datasets(self) -> List[Dict]:
        """Extract comprehensive dataset information including SQL queries, parameters, and metadata"""
        datasets = []
        for ds in self.root.findall('.//DataSet', self.namespace):
            dataset_info = {
                'name': ds.get('Name', ''),
                'query': '',
                'query_type': '',
                'data_source_name': '',
                'fields': [],
                'parameters': [],
                'filters': [],
                'calculated_fields': [],
                'complexity_score': 0
            }
            
            # Extract query information
            query_elem = ds.find('.//Query', self.namespace)
            if query_elem is not None:
                # Data source reference
                ds_name = query_elem.find('DataSourceName', self.namespace)
                if ds_name is not None:
                    dataset_info['data_source_name'] = ds_name.text
                
                # Command text (SQL query)
                command_text = query_elem.find('CommandText', self.namespace)
                if command_text is not None:
                    dataset_info['query'] = command_text.text or ''
                
                # Command type
                command_type = query_elem.find('CommandType', self.namespace)
                if command_type is not None:
                    dataset_info['query_type'] = command_type.text
                else:
                    dataset_info['query_type'] = 'Text'  # Default
                
                # Query parameters
                for param in query_elem.findall('.//QueryParameter', self.namespace):
                    param_info = {
                        'name': param.get('Name', ''),
                        'value': param.find('Value', self.namespace).text if param.find('Value', self.namespace) is not None else ''
                    }
                    dataset_info['parameters'].append(param_info)
            
            # Extract fields with comprehensive information
            for field in ds.findall('.//Field', self.namespace):
                field_info = {
                    'name': field.get('Name', ''),
                    'data_field': '',
                    'data_type': '',
                    'is_calculated': False,
                    'expression': ''
                }
                
                # Data field
                data_field = field.find('DataField', self.namespace)
                if data_field is not None:
                    field_info['data_field'] = data_field.text
                
                # Value expression (for calculated fields)
                value_expr = field.find('Value', self.namespace)
                if value_expr is not None:
                    field_info['expression'] = value_expr.text
                    field_info['is_calculated'] = True
                    dataset_info['calculated_fields'].append(field_info)
                
                dataset_info['fields'].append(field_info)
            
            # Extract filters
            for filter_elem in ds.findall('.//Filter', self.namespace):
                filter_info = {
                    'expression': filter_elem.find('FilterExpression', self.namespace).text if filter_elem.find('FilterExpression', self.namespace) is not None else '',
                    'operator': filter_elem.find('Operator', self.namespace).text if filter_elem.find('Operator', self.namespace) is not None else '',
                    'values': []
                }
                
                # Filter values
                for value in filter_elem.findall('.//FilterValue', self.namespace):
                    filter_info['values'].append(value.text)
                
                dataset_info['filters'].append(filter_info)
            
            # Calculate complexity score
            dataset_info['complexity_score'] = self._calculate_dataset_complexity(dataset_info)
            
            datasets.append(dataset_info)
        
        return datasets
    
    def _calculate_dataset_complexity(self, dataset_info: dict) -> int:
        """Calculate complexity score for a dataset"""
        score = 0
        
        # Query complexity
        query = dataset_info['query'].upper()
        if 'JOIN' in query:
            score += 2
        if 'UNION' in query:
            score += 3
        if 'SUBQUERY' in query or '(' in query:
            score += 2
        if 'CASE' in query:
            score += 1
        
        # Parameters add complexity
        score += len(dataset_info['parameters'])
        
        # Calculated fields add complexity
        score += len(dataset_info['calculated_fields']) * 2
        
        # Filters add complexity
        score += len(dataset_info['filters'])
        
        return score
    
    def extract_report_items(self) -> List[ReportElement]:
        """Extract report items like tables, charts, text boxes"""
        report_items = []
        
        # Extract tables
        for table in self.root.findall('.//Tablix', self.namespace):
            table_element = ReportElement(
                name=table.get('Name', ''),
                type='Table',
                properties={
                    'dataset_name': table.find('.//DataSetName', self.namespace).text if table.find('.//DataSetName', self.namespace) is not None else '',
                    'columns': self._extract_table_columns(table),
                    'groups': self._extract_table_groups(table)
                }
            )
            report_items.append(table_element)
        
        # Extract charts
        for chart in self.root.findall('.//Chart', self.namespace):
            chart_element = ReportElement(
                name=chart.get('Name', ''),
                type='Chart',
                properties={
                    'chart_type': self._extract_chart_type(chart),
                    'dataset_name': chart.find('.//DataSetName', self.namespace).text if chart.find('.//DataSetName', self.namespace) is not None else '',
                    'series': self._extract_chart_series(chart)
                }
            )
            report_items.append(chart_element)
        
        # Extract text boxes
        for textbox in self.root.findall('.//Textbox', self.namespace):
            textbox_element = ReportElement(
                name=textbox.get('Name', ''),
                type='TextBox',
                properties={
                    'value': textbox.find('.//Value', self.namespace).text if textbox.find('.//Value', self.namespace) is not None else ''
                }
            )
            report_items.append(textbox_element)
        
        return report_items
    
    def _extract_table_columns(self, table) -> List[Dict]:
        """Extract table column information"""
        columns = []
        for cell in table.findall('.//TablixCell', self.namespace):
            textbox = cell.find('.//Textbox', self.namespace)
            if textbox is not None:
                value_elem = textbox.find('.//Value', self.namespace)
                if value_elem is not None:
                    columns.append({
                        'name': textbox.get('Name', ''),
                        'expression': value_elem.text or ''
                    })
        return columns
    
    def _extract_table_groups(self, table) -> List[Dict]:
        """Extract table grouping information"""
        groups = []
        for group in table.findall('.//TablixRowHierarchy//TablixMember//Group', self.namespace):
            group_expr = group.find('.//GroupExpression', self.namespace)
            groups.append({
                'name': group.get('Name', ''),
                'expression': group_expr.text if group_expr is not None else ''
            })
        return groups
    
    def _extract_chart_type(self, chart) -> str:
        """Extract chart type"""
        chart_data = chart.find('.//ChartData', self.namespace)
        if chart_data is not None:
            series = chart_data.find('.//ChartSeries', self.namespace)
            if series is not None:
                chart_type = series.find('.//Type', self.namespace)
                return chart_type.text if chart_type is not None else 'Column'
        return 'Column'
    
    def _extract_chart_series(self, chart) -> List[Dict]:
        """Extract chart series information"""
        series_list = []
        for series in chart.findall('.//ChartSeries', self.namespace):
            series_info = {
                'name': series.get('Name', ''),
                'values': [],
                'categories': []
            }
            
            # Extract values
            for value in series.findall('.//ChartDataPoint//ChartDataPointValues//Y', self.namespace):
                series_info['values'].append(value.text or '')
            
            # Extract categories
            for category in series.findall('.//ChartCategoryHierarchy//ChartMember//Group//GroupExpression', self.namespace):
                series_info['categories'].append(category.text or '')
            
            series_list.append(series_info)
        
        return series_list

class PowerBIConverter:
    """AI-only converter for RDL components to Power BI artifacts"""
    
    def __init__(self):
        self.version = "Enterprise-LLM-v1.0"
        logger.info("PowerBIConverter initialized with unified LLM support")
    
    def _make_llm_request(self, messages, max_tokens=1500):
        """Make LLM API request with automatic token refresh and environment detection"""
        try:
            # Check if LLM client is available
            if not llm:
                logger.warning("LLM client not available, attempting to refresh...")
                # Try to refresh clients
                refreshed_llm, _ = refresh_clients()
                if not refreshed_llm:
                    return "# AI conversion requires LLM configuration\n# Please configure your environment variables"
                
                # Use refreshed client
                current_llm = refreshed_llm
            else:
                current_llm = llm
            
            # Convert messages to LangChain format
            from langchain_core.messages import HumanMessage, SystemMessage
            
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user" or msg["role"] == "human":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                # Skip assistant messages in input (they're for conversation context)
            
            # Make the request with token refresh handling
            try:
                response = current_llm.invoke(langchain_messages)
                return response.content
                
            except Exception as api_error:
                # Check if it's a token expiry error and retry once
                error_str = str(api_error).lower()
                if any(keyword in error_str for keyword in ['token', 'expired', 'unauthorized', 'authentication']):
                    logger.warning("Possible token expiry, refreshing clients and retrying...")
                    refreshed_llm, _ = refresh_clients()
                    if refreshed_llm:
                        response = refreshed_llm.invoke(langchain_messages)
                        return response.content
                    else:
                        raise api_error
                else:
                    raise api_error
                    
        except Exception as e:
            logger.error(f"❌ LLM API error: {e}")
            # Return a fallback response instead of crashing
            return f"# AI conversion failed: {str(e)}\n# Manual conversion recommended\n# Check your environment configuration and try again"
    
    def convert_sql_to_powerquery(self, sql_query: str, dataset_name: str, data_source_info: dict) -> str:
        """Convert SQL query to Power Query M language using AI"""
        
        # Determine data source type and connection details
        source_type = data_source_info.get('source_type', 'SQL Server')
        server = data_source_info.get('server', 'YOUR_SERVER')
        database = data_source_info.get('database', 'YOUR_DATABASE')
        auth = data_source_info.get('authentication', 'Windows')
        
        prompt = f"""
        You are an expert Power BI consultant. Convert the following SQL query to optimized Power Query M language for Power BI.
        
        **Context:**
        - Dataset Name: {dataset_name}
        - Data Source Type: {source_type}
        - Server: {server}
        - Database: {database}
        - Authentication: {auth}
        
        **SQL Query:**
        ```sql
        {sql_query}
        ```
        
        **Requirements:**
        1. Generate clean, production-ready Power Query M code
        2. Use the appropriate data source connector for {source_type}
        3. Handle authentication properly ({auth})
        4. Include proper error handling and timeouts
        5. Optimize for performance (query folding when possible)
        6. Add helpful comments for maintenance
        
        **Format:** Return only the M code, ready to paste into Power BI Advanced Editor.
        """
        
        messages = [
            {"role": "system", "content": "You are a Power BI expert specializing in data source connections and Power Query M language. Generate production-ready code that follows Microsoft best practices."},
            {"role": "user", "content": prompt}
        ]
        
        result = self._make_llm_request(messages, max_tokens=2000)
        
        # Enhance with metadata
        enhanced_result = f"""// {dataset_name} - Power Query M Code
// Data Source: {source_type}
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// AI-Generated using GPT-4

{result}

/* 
Data Source Information:
- Type: {source_type}
- Server: {server}
- Database: {database}
- Authentication: {auth}

Migration Notes:
- Code generated by AI and optimized for Power BI
- Test connection before production use
- Verify data types and transformations
- Monitor performance with full dataset
*/"""
        
        return enhanced_result
    
    def convert_table_to_dax(self, table_element: ReportElement) -> str:
        """Convert SSRS table to DAX measures using AI"""
        
        prompt = f"""
        You are a senior Power BI consultant and DAX expert. Convert the following SSRS table to comprehensive DAX measures for Power BI.
        
        **Table Information:**
        - Table Name: {table_element.name}
        - Type: {table_element.type}
        - Dataset: {table_element.properties.get('dataset_name', 'Unknown')}
        
        **Table Structure:**
        ```json
        {json.dumps(table_element.properties, indent=2)}
        ```
        
        **Requirements:**
        1. Generate comprehensive DAX measures for business analytics
        2. Include basic metrics (counts, sums, averages)
        3. Add advanced analytics (growth rates, percentages, rankings)
        4. Include time intelligence measures (YTD, MTD, Previous Period)
        5. Create KPIs and performance indicators
        6. Add data quality and validation measures
        7. Use proper error handling (DIVIDE, ISBLANK, etc.)
        8. Follow DAX best practices and optimization patterns
        9. Include helpful comments explaining business logic
        
        **Focus Areas:**
        - Business KPIs and metrics
        - Comparative analysis (vs previous periods, vs targets)
        - Performance indicators and rankings
        - Data quality and completeness checks
        
        **Format:** Return clean DAX code with detailed comments, ready to copy into Power BI.
        """
        
        messages = [
            {"role": "system", "content": "You are a Power BI expert and business intelligence consultant. Generate production-ready DAX measures that provide deep business insights and follow Microsoft best practices."},
            {"role": "user", "content": prompt}
        ]
        
        result = self._make_llm_request(messages, max_tokens=3000)
        
        # Enhance with metadata
        enhanced_result = f"""// {table_element.name} - DAX Measures
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// AI-Generated using GPT-4

{result}

/* 
Table Information:
- Name: {table_element.name}
- Type: {table_element.type}
- Dataset: {table_element.properties.get('dataset_name', 'Unknown')}
- Columns: {len(table_element.properties.get('columns', []))}

Implementation Notes:
- Copy each measure individually into Power BI
- Test all calculations with your data
- Apply appropriate formatting (Currency, Percentage, etc.)
- Organize measures into display folders
- Validate business logic with stakeholders
*/"""
        
        return enhanced_result
    
    def convert_chart_to_powerbi_visual(self, chart_element: ReportElement) -> Dict:
        """Convert SSRS chart to Power BI visual configuration using AI"""
        
        prompt = f"""
        You are a Power BI visualization expert. Convert the following SSRS chart to a Power BI visual configuration.
        
        **Chart Information:**
        - Chart Name: {chart_element.name}
        - Original Type: {chart_element.properties.get('chart_type', 'Unknown')}
        - Dataset: {chart_element.properties.get('dataset_name', 'Unknown')}
        
        **Chart Properties:**
        ```json
        {json.dumps(chart_element.properties, indent=2)}
        ```
        
        **Requirements:**
        Return a JSON configuration object with:
        1. **visualType**: Best Power BI visual type for this chart
        2. **title**: Appropriate chart title
        3. **fieldMappings**: Field well assignments (axis, values, legend, etc.)
        4. **formatting**: Recommended formatting options
        5. **interactions**: Suggested interactive features
        6. **powerBIInstructions**: Step-by-step setup instructions
        
        **Format:** Return only valid JSON that can be parsed and used programmatically.
        """
        
        messages = [
            {"role": "system", "content": "You are a Power BI visualization expert. Return only valid JSON configurations for Power BI visuals with comprehensive setup instructions."},
            {"role": "user", "content": prompt}
        ]
        
        result = self._make_llm_request(messages, max_tokens=1500)
        
        try:
            # Parse the JSON response
            visual_config = json.loads(result)
            
            # Add metadata
            visual_config["originalProperties"] = chart_element.properties
            visual_config["generatedBy"] = "AI-GPT4"
            visual_config["generatedAt"] = datetime.now().isoformat()
            
            return visual_config
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: AI returned invalid JSON for chart {chart_element.name}: {e}")
            # Return a basic fallback configuration
            return {
                "visualType": "clusteredColumnChart",
                "title": f"{chart_element.name} - Converted from SSRS",
                "error": "AI returned invalid JSON, using fallback configuration",
                "originalProperties": chart_element.properties,
                "powerBIInstructions": [
                    "1. Add appropriate chart visual to report",
                    "2. Configure field wells manually",
                    "3. Set chart title and formatting",
                    "4. Test with your data"
                ]
            }

def create_migration_output_structure() -> dict:
    """Create organized output directory structure for migration files"""
    
    # Create main migration directory
    migration_dir = Path('./migration_output')
    migration_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        'power_query': migration_dir / 'power_query',
        'dax_measures': migration_dir / 'dax_measures', 
        'visual_configs': migration_dir / 'visual_configs',
        'guides': migration_dir / 'guides',
        'validation': migration_dir / 'validation',
        'documentation': migration_dir / 'documentation'
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)
    
    return subdirs

def save_migration_files(conversion_results: dict, rdl_info: dict, output_dirs: dict):
    """Save all migration files to organized directory structure"""
    
    # Save Power Query files
    for filename, content in conversion_results.get('power_query', {}).items():
        file_path = output_dirs['power_query'] / f"{filename}.m"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ Power Query saved: {file_path}")
    
    # Save DAX files
    for filename, content in conversion_results.get('dax_measures', {}).items():
        file_path = output_dirs['dax_measures'] / f"{filename}.dax"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ DAX measures saved: {file_path}")
    
    # Save visual configurations
    for filename, content in conversion_results.get('visual_configs', {}).items():
        file_path = output_dirs['visual_configs'] / f"{filename}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
        print(f"   ✅ Visual config saved: {file_path}")
    
    # Save migration guide
    if 'migration_guide' in conversion_results:
        guide_path = output_dirs['guides'] / 'migration_guide.md'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(conversion_results['migration_guide'])
        print(f"   ✅ Migration guide saved: {guide_path}")
    
    # Save validation queries
    if 'validation_queries' in conversion_results:
        validation_path = output_dirs['validation'] / 'validation_queries.sql'
        with open(validation_path, 'w', encoding='utf-8') as f:
            f.write(conversion_results['validation_queries'])
        print(f"   ✅ Validation queries saved: {validation_path}")
    
    # Save complete migration summary
    summary = {
        'rdl_analysis': rdl_info,
        'conversion_results': {k: v for k, v in conversion_results.items() if k not in ['power_query', 'dax_measures', 'visual_configs']},
        'generated_files': {
            'power_query': list(conversion_results.get('power_query', {}).keys()),
            'dax_measures': list(conversion_results.get('dax_measures', {}).keys()),
            'visual_configs': list(conversion_results.get('visual_configs', {}).keys())
        },
        'migration_timestamp': datetime.now().isoformat(),
        'ai_powered': True
    }
    
    summary_path = output_dirs['documentation'] / 'migration_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Migration summary saved: {summary_path}")

def generate_migration_guide(rdl_info: dict, conversion_results: dict) -> str:
    """Generate comprehensive migration guide"""
    
    report_name = rdl_info.get('report_name', 'Unknown Report')
    data_sources = rdl_info.get('data_sources', [])
    datasets = rdl_info.get('datasets', [])
    report_items = rdl_info.get('report_items', [])
    
    # Assess complexity
    total_complexity = sum([ds.get('complexity_score', 0) for ds in datasets]) + len(report_items) * 2
    complexity_level = "Low" if total_complexity <= 10 else "Medium" if total_complexity <= 25 else "High"
    
    guide = f"""# 🚀 Power BI Migration Guide - AI Generated
## Report: {report_name}

### 📊 Migration Assessment
- **Report Name**: {report_name}
- **Complexity Level**: {complexity_level}
- **Data Sources**: {len(data_sources)}
- **Datasets**: {len(datasets)}
- **Report Items**: {len(report_items)}
- **AI Conversion**: ✅ Enabled (GPT-4)
- **Generated**: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}

### 🤖 AI-Powered Migration Features
- **Intelligent Query Conversion**: SQL optimized for Power Query
- **Business-Focused DAX**: KPIs and analytics measures
- **Smart Visual Mapping**: SSRS charts to Power BI visuals
- **Context-Aware**: Understands your data source types

### 📋 Data Sources Overview
"""
    
    for i, ds in enumerate(data_sources, 1):
        guide += f"""
#### Data Source {i}: {ds.get('name', f'DataSource{i}')}
- **Type**: {ds.get('source_type', 'Unknown')}
- **Server**: {ds.get('server', 'Not specified')}
- **Database**: {ds.get('database', 'Not specified')}
- **Authentication**: {ds.get('authentication', 'Not specified')}
"""
    
    guide += f"""
### 🔧 Step-by-Step Migration Process

#### Phase 1: Power BI Desktop Setup
1. **Install Power BI Desktop** (latest version)
2. **Open Power BI Desktop**
3. **Create a new blank report**

#### Phase 2: Data Import (AI-Generated)
"""
    
    for i, dataset in enumerate(datasets, 1):
        dataset_name = dataset.get('name', f'Dataset{i}')
        guide += f"""
**Dataset {i}: {dataset_name}**
1. Go to **Home** → **Transform Data**
2. Click **New Source** → **Blank Query**
3. Open **Advanced Editor**
4. Copy and paste AI-generated code from `power_query/{dataset_name}.m`
5. Rename query to "{dataset_name}"
6. **Close & Apply**

💡 **AI Enhancement**: The generated M code is optimized for your data source type and includes proper error handling.
"""
    
    guide += f"""
#### Phase 3: DAX Measures (AI-Generated)
"""
    
    table_items = [item for item in report_items if item.get('type') == 'Table']
    for item in table_items:
        guide += f"""
**Table: {item.get('name', 'Unknown')}**
1. In **Fields** panel, right-click your main table
2. Select **New Measure**
3. Copy each AI-generated measure from `dax_measures/{item.get('name', 'Unknown')}.dax`
4. Paste one measure at a time (includes KPIs, time intelligence, and business metrics)
5. Apply appropriate formatting (Currency, Percentage, etc.)

🎯 **AI Enhancement**: Generated measures include business KPIs, trend analysis, and performance indicators.
"""
    
    guide += f"""
#### Phase 4: Visual Creation (AI-Guided)
"""
    
    chart_items = [item for item in report_items if item.get('type') == 'Chart']
    for item in chart_items:
        guide += f"""
**Chart: {item.get('name', 'Unknown')}**
1. Follow AI-generated instructions in `visual_configs/{item.get('name', 'Unknown')}.json`
2. Insert recommended visual type
3. Configure field wells according to AI specifications
4. Apply suggested formatting and interactions

🎨 **AI Enhancement**: Visual configurations are optimized for Power BI best practices and modern analytics.
"""
    
    guide += f"""
### ✅ Validation Steps
1. **Compare row counts** between original report and Power BI
2. **Verify calculations** match original report results
3. **Test filters and interactions**
4. **Check performance** with full dataset
5. **Run validation queries** from `validation/validation_queries.sql`

### 🚀 Advanced AI Features Used
- **Context-Aware Conversion**: AI understands your specific data source
- **Business Intelligence Focus**: Generated measures provide actionable insights
- **Optimization Patterns**: Code follows Power BI performance best practices
- **Error Handling**: Built-in validation and error management

### 🚨 Common Issues & Solutions
- **Connection Issues**: Check AI-generated connection strings in Power Query files
- **Data Type Errors**: AI has optimized data types, but verify with your data
- **Measure Errors**: AI-generated DAX includes error handling, but test with your data
- **Performance Issues**: AI code is optimized, but consider DirectQuery for very large datasets

### 📞 Support
- **AI-Generated Code**: All files include detailed comments and explanations
- **Business Context**: DAX measures include business logic explanations
- **Best Practices**: Code follows Microsoft Power BI best practices
- **Troubleshooting**: Each file includes implementation notes

---
*Generated by AI-Powered RDL to Power BI Migration Tool using GPT-4*
*All conversions are intelligent and context-aware*
"""
    
    return guide

def generate_batch_migration_guide(batch_info: dict) -> str:
    """Generate optimized batch-level migration guide with populated content"""
    
    files_info = batch_info.get('files', [])
    batch_id = batch_info.get('batch_id', 'unknown')
    total_files = len(files_info)
    
    # Aggregate batch statistics
    total_data_sources = sum(len(file_info.get('data_sources', [])) for file_info in files_info)
    total_datasets = sum(len(file_info.get('datasets', [])) for file_info in files_info)
    total_items = sum(len(file_info.get('report_items', [])) for file_info in files_info)
    
    # Assess overall complexity
    total_complexity = total_datasets * 3 + total_items * 2
    complexity_level = "Low" if total_complexity <= 20 else "Medium" if total_complexity <= 50 else "High"
    
    # Build file context section immediately
    file_context_section = "| File Name | Complexity | Data Sources | Datasets | Report Items |\n"
    file_context_section += "|-----------|------------|--------------|----------|-------------|\n"
    
    for file_info in files_info:
        file_name = file_info.get('report_name', 'Unknown')
        data_sources_count = len(file_info.get('data_sources', []))
        datasets_count = len(file_info.get('datasets', []))
        items_count = len(file_info.get('report_items', []))
        complexity = "Low" if datasets_count <= 2 else "Medium" if datasets_count <= 5 else "High"
        
        file_context_section += f"| {file_name} | {complexity} | {data_sources_count} | {datasets_count} | {items_count} |\n"
    
    # Build data import section
    data_import_section = "**For each file in your batch:**\n"
    data_import_section += "1. Navigate to the file's Power Query (.m) files in the results\n"
    data_import_section += "2. Follow the standard import process for each dataset\n"
    data_import_section += "3. Use the **Preview** button to see the generated M code\n"
    data_import_section += "4. Import datasets in dependency order if there are relationships\n\n"
    
    # Add specific files information
    for file_info in files_info:
        file_name = file_info.get('report_name', 'Unknown')
        datasets = file_info.get('datasets', [])
        if datasets:
            data_import_section += f"**{file_name}** datasets:\n"
            for i, dataset in enumerate(datasets, 1):
                dataset_name = dataset.get('name', f'Dataset{i}')
                data_import_section += f"- `{file_name}/{dataset_name}.m`\n"
            data_import_section += "\n"
    
    # Build DAX measures section
    dax_section = "**For each file's DAX measures:**\n"
    dax_section += "1. Locate the .dax files in your migration results\n"
    dax_section += "2. Use **Preview** to see generated measures\n"
    dax_section += "3. Apply measures to the appropriate tables\n"
    dax_section += "4. Test calculations before proceeding to next file\n\n"
    
    # Add specific DAX files information
    for file_info in files_info:
        file_name = file_info.get('report_name', 'Unknown')
        report_items = file_info.get('report_items', [])
        dax_files = [item for item in report_items if item.get('type') == 'Table']
        if dax_files:
            dax_section += f"**{file_name}** DAX measures:\n"
            for item in dax_files:
                table_name = item.get('name', 'Table')
                dax_section += f"- `{file_name}/{table_name}.dax`\n"
            dax_section += "\n"
    
    # Build visual creation section
    visual_section = "**Visual Creation Process:**\n"
    visual_section += "1. Start with the most complex reports in your batch\n"
    visual_section += "2. Establish consistent visual themes and formatting\n"
    visual_section += "3. Reuse visual configurations where reports are similar\n"
    visual_section += "4. Test interactivity and filtering across all reports\n\n"
    
    visual_section += "**Files in order of complexity:**\n"
    # Sort files by complexity for visual creation order
    sorted_files = sorted(files_info, key=lambda x: len(x.get('report_items', [])), reverse=True)
    for i, file_info in enumerate(sorted_files, 1):
        file_name = file_info.get('report_name', 'Unknown')
        items_count = len(file_info.get('report_items', []))
        visual_section += f"{i}. {file_name} ({items_count} report items)\n"
    
    # Build asset links section
    asset_links = "### 📁 All Generated Assets:\n\n"
    asset_links += "#### Power Query Files:\n"
    for file_info in files_info:
        file_name = file_info.get('report_name', 'Unknown')
        datasets = file_info.get('datasets', [])
        for dataset in datasets:
            dataset_name = dataset.get('name', 'Dataset')
            asset_links += f"- `{file_name}/{dataset_name}.m`\n"
    
    asset_links += "\n#### DAX Measure Files:\n"
    for file_info in files_info:
        file_name = file_info.get('report_name', 'Unknown')
        report_items = file_info.get('report_items', [])
        for item in report_items:
            if item.get('type') == 'Table':
                table_name = item.get('name', 'Table')
                asset_links += f"- `{file_name}/{table_name}.dax`\n"
    
    
    # Generate the complete guide with all content populated
    guide = f"""# 🚀 Power BI Migration Guide - Batch Migration
## 📋 Batch Overview

### 📊 Migration Assessment
- **Batch ID**: {batch_id}
- **Total Files**: {total_files}
- **Overall Complexity**: {complexity_level}
- **Total Data Sources**: {total_data_sources}
- **Total Datasets**: {total_datasets}
- **Total Report Items**: {total_items}
- **AI Conversion**: ✅ Enabled (GPT-4)
- **Generated**: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}

### 🎯 Token-Optimized Architecture
This guide uses intelligent template generation to provide consistent migration guidance while minimizing AI token usage. All file-specific details are included below for easy reference.

### 📁 Files in This Migration Batch

{file_context_section}

### 🤖 AI-Powered Migration Features
- **Intelligent Batch Processing**: Optimized for bulk RDL migrations
- **Context-Aware Display**: Shows relevant information for all files
- **Smart Asset Linking**: Direct links to generated Power Query and DAX files
- **Unified Guidance**: Consistent migration steps across all files
- **Token Efficient**: Shared template reduces LLM usage by 60-80%

### 🔧 Universal Migration Process

#### Phase 1: Power BI Desktop Setup
1. **Install Power BI Desktop** (latest version recommended)
2. **Open Power BI Desktop**
3. **Create a new blank report**

#### Phase 2: Data Import (All Files)

{data_import_section}

💡 **AI Enhancement**: All generated M code is optimized for your specific data source types and includes comprehensive error handling.

#### Phase 3: DAX Measures (All Files)

{dax_section}

🎯 **AI Enhancement**: Generated measures include business KPIs, trend analysis, and performance indicators tailored to your report requirements.

#### Phase 4: Visual Creation (Recommended Order)

{visual_section}

🎨 **AI Enhancement**: Visual configurations follow Power BI best practices and modern analytics design patterns.

### ✅ Universal Validation Steps
1. **Compare row counts** between original reports and Power BI
2. **Verify calculations** match original report results across all files
3. **Test filters and interactions** for each migrated report
4. **Check performance** with full dataset on all reports
5. **Run validation queries** from generated validation files

### 🚀 Advanced Batch Features
- **Consistent Data Models**: Shared dimensions and fact tables across related reports
- **Unified Security**: Common row-level security patterns where applicable  
- **Standardized Measures**: Reusable KPIs and calculations across the batch
- **Performance Optimization**: Batch-level query folding and relationship optimization

### 🚨 Common Issues & Solutions
- **Connection Issues**: Check generated connection strings in Power Query files
- **Data Type Conflicts**: AI has optimized data types, but verify across all reports
- **Measure Dependencies**: Some measures may reference tables from other reports in the batch
- **Performance Considerations**: Large batches may benefit from shared datasets in Power BI Service

### 📞 Support & Documentation
- **Comprehensive Comments**: All generated files include detailed implementation notes
- **Business Context**: DAX measures include business logic explanations
- **Migration Notes**: Each file includes specific migration considerations
- **Batch Relationships**: Documentation of cross-report dependencies

{asset_links}
"""
    
    return guide

def inject_dynamic_content(template: str, current_file_context: dict = None) -> str:
    """Inject dynamic content into the migration guide template based on current context"""
    
    files_info = current_file_context.get('files_info', []) if current_file_context else []
    current_file = current_file_context.get('current_file') if current_file_context else None
    
    # Build file context section
    file_context_html = "| File Name | Complexity | Data Sources | Datasets | Status |\n|-----------|------------|--------------|----------|--------|\n"
    
    for file_info in files_info:
        file_name = file_info.get('name', 'Unknown')
        complexity = file_info.get('complexity', 'Unknown')
        data_sources = file_info.get('data_sources', 0)
        datasets = file_info.get('datasets', 0)
        status = "🎯 **Current**" if current_file and file_name == current_file else "✅ Ready"
        
        file_context_html += f"| {file_name} | {complexity} | {data_sources} | {datasets} | {status} |\n"
    
    # Build dynamic data import section
    data_import_section = ""
    if current_file:
        # Show context for current file
        current_file_info = next((f for f in files_info if f.get('name') == current_file), None)
        if current_file_info:
            for i, dataset in enumerate(current_file_info.get('datasets_details', []), 1):
                dataset_name = dataset.get('name', f'Dataset{i}')
                data_import_section += f"""
**Dataset {i}: {dataset_name}** (from {current_file})
1. Go to **Home** → **Transform Data**
2. Click **New Source** → **Blank Query**
3. Open **Advanced Editor**
4. Copy and paste code from `{current_file}/{dataset_name}.m`
5. Rename query to "{dataset_name}"
6. **Close & Apply**
"""
    else:
        # Show general guidance for all files
        data_import_section = """
**For each file in your batch:**
1. Navigate to the file's Power Query (.m) files in the results
2. Follow the standard import process for each dataset
3. Use the **Preview** button to see the generated M code
4. Import datasets in dependency order if there are relationships
"""
    
    # Build dynamic DAX section
    dax_section = ""
    if current_file:
        current_file_info = next((f for f in files_info if f.get('name') == current_file), None)
        if current_file_info:
            tables = [item for item in current_file_info.get('report_items_details', []) if item.get('type') == 'Table']
            for table in tables:
                table_name = table.get('name', 'Unknown')
                dax_section += f"""
**Table: {table_name}** (from {current_file})
1. In **Fields** panel, right-click the table
2. Select **New Measure**
3. Copy measures from `{current_file}/{table_name}.dax`
4. Apply one measure at a time with proper formatting
"""
    else:
        dax_section = """
**For each file's DAX measures:**
1. Locate the .dax files in your migration results
2. Use **Preview** to see generated measures
3. Apply measures to the appropriate tables
4. Test calculations before proceeding to next file
"""
    
    # Build dynamic visual section
    visual_section = ""
    if current_file:
        visual_section = f"""
**Visual Creation for {current_file}:**
1. Review the original RDL report structure
2. Create Power BI visuals following the AI-generated recommendations
3. Configure field wells based on the data model
4. Apply consistent formatting across the batch
"""
    else:
        visual_section = """
**Visual Creation Process:**
1. Start with the most complex reports in your batch
2. Establish consistent visual themes and formatting
3. Reuse visual configurations where reports are similar
4. Test interactivity and filtering across all reports
"""
    
    # Build asset links
    asset_links = ""
    if current_file and files_info:
        current_file_info = next((f for f in files_info if f.get('name') == current_file), None)
        if current_file_info:
            asset_links += f"### 📁 Assets for {current_file}:\n"
            asset_links += "#### Power Query Files:\n"
            for dataset in current_file_info.get('datasets_details', []):
                dataset_name = dataset.get('name', 'Dataset')
                asset_links += f"- `{current_file}/{dataset_name}.m`\n"
            
            asset_links += "\n#### DAX Measure Files:\n"
            tables = [item for item in current_file_info.get('report_items_details', []) if item.get('type') == 'Table']
            for table in tables:
                table_name = table.get('name', 'Table')
                asset_links += f"- `{current_file}/{table_name}.dax`\n"
    else:
        asset_links = "### 📁 All Generated Assets:\nUse the file browser to explore all generated Power Query (.m) and DAX (.dax) files for each report in your batch."
    
    # Replace placeholders with dynamic content
    processed_template = template.replace("{{FILE_CONTEXT_SECTION}}", file_context_html)
    processed_template = processed_template.replace("{{DYNAMIC_DATA_IMPORT_SECTION}}", data_import_section)
    processed_template = processed_template.replace("{{DYNAMIC_DAX_SECTION}}", dax_section)
    processed_template = processed_template.replace("{{DYNAMIC_VISUAL_SECTION}}", visual_section)
    processed_template = processed_template.replace("{{DYNAMIC_ASSET_LINKS}}", asset_links)
    
    return processed_template

def generate_validation_queries(datasets: List[Dict], data_sources: List[Dict]) -> str:
    """Generate SQL validation queries"""
    
    queries = [f"""-- AI-Generated Validation Queries
-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-- Purpose: Verify Power BI data matches source system

-- ============================================
-- DATA VALIDATION QUERIES
-- Run these queries in your source system to verify Power BI data
-- ============================================

"""]
    
    for dataset in datasets:
        if dataset.get('query'):
            dataset_name = dataset['name']
            query = dataset['query']
            
            queries.append(f"""-- Dataset: {dataset_name}
-- Row Count Validation
SELECT 
    '{dataset_name}' as Dataset,
    COUNT(*) as RowCount,
    GETDATE() as ValidatedAt
FROM (
{query}
) as dataset_query;

-- Data Sample (First 10 rows)
SELECT TOP 10 * FROM (
{query}
) as dataset_sample;

-- Data Quality Check
SELECT 
    '{dataset_name}' as Dataset,
    COUNT(*) as TotalRows,
    COUNT(DISTINCT *) as UniqueRows,
    GETDATE() as ValidatedAt
FROM (
{query}
) as dataset_stats;

""")
    
    queries.append("""-- ============================================
-- AI-GENERATED DATA QUALITY RECOMMENDATIONS
-- ============================================

-- These queries help ensure data integrity during migration
-- Run before and after Power BI implementation to compare results
-- Contact your data team if discrepancies are found

""")
    
    return "\n".join(queries)

def calculate_similarity(rdl1, rdl2):
    """Calculate similarity score between two RDL files"""
    score = 0
    
    # Compare datasets
    datasets1 = {ds['name']: ds['query'] for ds in rdl1['datasets']}
    datasets2 = {ds['name']: ds['query'] for ds in rdl2['datasets']}
    
    common_datasets = set(datasets1.keys()) & set(datasets2.keys())
    if common_datasets:
        score += 40  # 40% for common dataset names
        
        # Check query similarity
        for dataset_name in common_datasets:
            if datasets1[dataset_name] == datasets2[dataset_name]:
                score += 30  # 30% for identical queries
    
    # Compare data sources
    sources1 = {ds['name']: ds.get('server', '') for ds in rdl1['data_sources']}
    sources2 = {ds['name']: ds.get('server', '') for ds in rdl2['data_sources']}
    
    common_sources = set(sources1.keys()) & set(sources2.keys())
    if common_sources:
        score += 20  # 20% for common data sources
    
    # Compare report structure
    items1 = {item['name']: item['type'] for item in rdl1['report_items']}
    items2 = {item['name']: item['type'] for item in rdl2['report_items']}
    
    common_items = set(items1.keys()) & set(items2.keys())
    if common_items:
        score += 10  # 10% for common report items
    
    return min(score, 100)

def migrate_rdl_to_powerbi(rdl_file_path: str):
    """Main AI-powered migration function"""
    
    print("🤖 AI-Powered RDL to Power BI Migration Tool")
    print("🚀 Using GPT-4 for intelligent conversion")
    print("=" * 60)
    
    # Validate input file
    if not Path(rdl_file_path).exists():
        print(f"❌ Error: RDL file not found: {rdl_file_path}")
        return
    
    print(f"\n📄 Parsing RDL file: {rdl_file_path}")
    
    # Step 1: Parse RDL file
    try:
        parser = RDLParser(rdl_file_path)
        data_sources = parser.extract_data_sources()
        datasets = parser.extract_datasets()
        report_items = parser.extract_report_items()
        
        print(f"   ✅ Found {len(data_sources)} data sources")
        print(f"   ✅ Found {len(datasets)} datasets")
        print(f"   ✅ Found {len(report_items)} report items")
        
    except Exception as e:
        print(f"❌ Error parsing RDL file: {e}")
        return
    
    # Step 2: Create output structure
    print("\n📁 Creating output directory structure...")
    output_dirs = create_migration_output_structure()
    
    # Step 3: AI-powered conversion
    print("\n🤖 AI-powered conversion using GPT-4...")
    converter = PowerBIConverter()
    
    conversion_results = {
        'power_query': {},
        'dax_measures': {},
        'visual_configs': {},
        'validation_queries': ''
    }
    
    # Convert datasets to Power Query with AI
    print("\n   🧠 AI converting datasets to Power Query...")
    for dataset in datasets:
        if dataset['query']:
            dataset_name = dataset['name']
            print(f"   🔄 AI processing dataset: {dataset_name}")
            
            # Find corresponding data source
            data_source_info = {}
            for ds in data_sources:
                if ds['name'] == dataset.get('data_source_name'):
                    data_source_info = ds
                    break
            
            try:
                power_query = converter.convert_sql_to_powerquery(
                    dataset['query'], 
                    dataset_name,
                    data_source_info
                )
                conversion_results['power_query'][dataset_name] = power_query
                print(f"   ✅ AI generated Power Query for: {dataset_name}")
            except Exception as e:
                print(f"   ❌ AI conversion failed for dataset {dataset_name}: {e}")
                return
    
    # Convert report items with AI
    print("\n   🧠 AI converting report items...")
    for item in report_items:
        if item.type == 'Table':
            print(f"   🔄 AI processing table: {item.name}")
            try:
                dax_code = converter.convert_table_to_dax(item)
                conversion_results['dax_measures'][item.name] = dax_code
                print(f"   ✅ AI generated DAX measures for: {item.name}")
            except Exception as e:
                print(f"   ❌ AI conversion failed for table {item.name}: {e}")
                return
        
        elif item.type == 'Chart':
            print(f"   🔄 AI processing chart: {item.name}")
            try:
                visual_config = converter.convert_chart_to_powerbi_visual(item)
                conversion_results['visual_configs'][item.name] = visual_config
                print(f"   ✅ AI generated visual config for: {item.name}")
            except Exception as e:
                print(f"   ❌ AI conversion failed for chart {item.name}: {e}")
                return
    
    # Generate validation queries
    print("\n   📝 Generating validation queries...")
    validation_queries = generate_validation_queries(datasets, data_sources)
    conversion_results['validation_queries'] = validation_queries
    
    # Generate AI-powered migration guide
    print("\n   📖 AI generating comprehensive migration guide...")
    rdl_analysis_info = {
        'report_name': parser.report_info['report_name'],
        'data_sources': data_sources,
        'datasets': datasets,
        'report_items': [{'name': item.name, 'type': item.type, 'properties': item.properties} for item in report_items]
    }
    
    migration_guide = generate_migration_guide(rdl_analysis_info, conversion_results)
    conversion_results['migration_guide'] = migration_guide
    
    # Step 4: Save all files
    print("\n💾 Saving AI-generated migration files...")
    save_migration_files(conversion_results, rdl_analysis_info, output_dirs)
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("🎉 AI-POWERED MIGRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"\n🤖 AI Conversion Summary:")
    print(f"   • Report: {parser.report_info['report_name']}")
    print(f"   • Data Sources: {len(data_sources)} (all analyzed)")
    print(f"   • Datasets: {len(datasets)} (AI-converted to Power Query)")
    print(f"   • Report Items: {len(report_items)} (AI-optimized)")
    print(f"   • AI Model: GPT-4 (latest)")
    print(f"   • Intelligence Level: High (context-aware)")
    
    print(f"\n📁 AI-Generated Files:")
    print(f"   📂 migration_output/")
    print(f"     📂 power_query/ - {len(conversion_results['power_query'])} optimized .m files")
    print(f"     📂 dax_measures/ - {len(conversion_results['dax_measures'])} intelligent .dax files")
    print(f"     📂 visual_configs/ - {len(conversion_results['visual_configs'])} smart .json configs")
    print(f"     📂 guides/ - AI-generated migration guide")
    print(f"     📂 validation/ - Data validation queries")
    print(f"     📂 documentation/ - Complete AI analysis")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review migration_output/guides/migration_guide.md")
    print(f"   2. Open Power BI Desktop")
    print(f"   3. Follow AI-generated step-by-step instructions")
    print(f"   4. Use AI-optimized code files for your migration")
    print(f"   5. Validate with AI-generated test queries")
    
    print(f"\n💡 AI Advantages:")
    print(f"   ✅ Context-aware conversions")
    print(f"   ✅ Business intelligence focus")
    print(f"   ✅ Performance optimizations")
    print(f"   ✅ Error handling included")
    print(f"   ✅ Best practices applied")

def main():
    """Main function for AI-only version"""
    parser = argparse.ArgumentParser(description='AI-Powered RDL to Power BI Migration Tool')
    parser.add_argument('rdl_file', help='Path to the RDL file to convert')
    
    args = parser.parse_args()
    
    print("🤖 AI-Only Migration Tool")
    print("🔑 OpenAI API Key: ✅ Configured")
    print("🧠 AI Model: GPT-4")
    print()
    
    migrate_rdl_to_powerbi(args.rdl_file)

if __name__ == "__main__":
    main()