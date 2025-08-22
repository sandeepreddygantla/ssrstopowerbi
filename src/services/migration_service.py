"""
RDL Business Logic Analyzer
==========================
Enterprise-grade similarity analysis based on business logic, not text similarity.
Designed specifically for SSRS RDL reports to identify true functional duplicates.
Enhanced with AI-powered query analysis for cost-effective similarity detection.
Supports multi-stage filtering to minimize token usage while maximizing accuracy.
"""

import xml.etree.ElementTree as ET
import re
import sqlparse
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging
from src.models.rdl_models import (
    FilterType, CalculationType, BusinessFilter, BusinessCalculation, 
    BusinessOutput, BusinessContext
)
from src.utils.rdl_parser import RDLParserHelpers
from src.services.ai_query_analyzer import AIQueryAnalyzer
from src.utils.embedding_manager import EmbeddingManager
from src.utils.query_preprocessor import QueryPreprocessor
from src.cache.similarity_cache import SimilarityCacheManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RDLBusinessAnalyzer:
    """
    Enhanced RDL analyzer with AI-powered query analysis.
    Provides cost-effective similarity detection with multi-stage filtering.
    Supports large-scale analysis (50+ files) with intelligent caching.
    """
    
    def __init__(self, enable_ai_analysis: bool = True, cache_enabled: bool = True):
        self.namespace_map = {
            'rdl': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition',
            'rd': 'http://schemas.microsoft.com/SQLServer/reporting/reportdesigner'
        }
        # Will be set dynamically based on actual file namespace
        
        # AI-powered components
        self.enable_ai_analysis = enable_ai_analysis
        self.ai_analyzer = AIQueryAnalyzer() if enable_ai_analysis else None
        self.embedding_manager = EmbeddingManager() if enable_ai_analysis else None
        self.query_preprocessor = QueryPreprocessor() if enable_ai_analysis else None
        
        # Caching system
        self.cache_manager = SimilarityCacheManager() if cache_enabled else None
        
        # Performance tracking
        self.analysis_stats = {
            'total_queries_analyzed': 0,
            'ai_analyses_performed': 0,
            'cache_hits': 0,
            'token_usage': 0,
            'cost_savings': 0.0
        }
    
    def _detect_namespace(self, root):
        """Detect the correct RDL namespace from the root element"""
        root_tag = root.tag
        if '}' in root_tag:
            namespace = root_tag.split('}')[0] + '}'
            # Update namespace mapping with detected namespace
            self.namespace_map['rdl'] = namespace.strip('{}')
            logger.info(f"Detected RDL namespace: {self.namespace_map['rdl']}")
        else:
            # If no namespace in root tag, check for default namespace in attributes
            default_ns = root.get('xmlns')
            if default_ns:
                self.namespace_map['rdl'] = default_ns
                logger.info(f"Using default namespace from xmlns attribute: {default_ns}")
            else:
                logger.warning("No namespace detected, using default namespace mapping")
    
    def analyze_rdl_file(self, file_path: str) -> BusinessContext:
        """Extract complete business context from RDL file"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Detect and set the correct namespace based on the actual file
            self._detect_namespace(root)
            
            # Extract all business components
            data_sources = self._extract_data_sources(root)
            filters = self._extract_business_filters(root)
            calculations = self._extract_calculations(root)
            output = self._extract_output_structure(root)
            parameters = self._extract_parameters(root)
            sql_queries = self._extract_sql_queries(root)  # Extract SQL queries for manual validation
            
            # Infer business purpose from structure
            business_purpose = self._infer_business_purpose(filters, calculations, output)
            
            context = BusinessContext(
                data_sources=data_sources,
                filters=filters,
                calculations=calculations,
                output=output,
                parameters=parameters,
                business_purpose=business_purpose
            )
            
            # Add SQL queries as additional attribute for manual validation
            context.sql_queries = sql_queries
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing RDL file {file_path}: {str(e)}")
            raise
    
    def _extract_sql_queries(self, root) -> List[Dict[str, str]]:
        """Extract all SQL queries from RDL DataSets for manual validation"""
        sql_queries = []
        
        try:
            # First, try with the detected namespace
            datasets = root.findall('.//rdl:DataSet', self.namespace_map)
            
            # If no datasets found, try without namespace (fallback)
            if not datasets:
                logger.info("No datasets found with namespace, trying without namespace")
                datasets = root.findall('.//DataSet')
            
            # If still no datasets, try to find with any namespace
            if not datasets:
                # Get the actual namespace from the root element
                root_ns = None
                if '}' in root.tag:
                    root_ns = root.tag.split('}')[0] + '}'
                if root_ns:
                    # Create a temporary namespace map with the detected namespace
                    temp_ns_map = {'ns': root_ns.strip('{}')}
                    datasets = root.findall('.//ns:DataSet', temp_ns_map)
                    logger.info(f"Searching with detected namespace: {root_ns}")
            
            logger.info(f"Found {len(datasets)} DataSet elements")
            
            for dataset in datasets:
                dataset_name = dataset.get('Name', 'Unknown')
                logger.info(f"Processing DataSet: {dataset_name}")
                
                # Find CommandText within Query - try multiple approaches
                command_text_elem = None
                
                # Method 1: With namespace
                command_text_elem = dataset.find('.//rdl:CommandText', self.namespace_map)
                
                # Method 2: Without namespace (fallback)
                if command_text_elem is None:
                    command_text_elem = dataset.find('.//CommandText')
                
                # Method 3: With detected namespace
                if command_text_elem is None and '}' in root.tag:
                    root_ns = root.tag.split('}')[0] + '}'
                    temp_ns_map = {'ns': root_ns.strip('{}')}
                    command_text_elem = dataset.find('.//ns:CommandText', temp_ns_map)
                
                if command_text_elem is not None and command_text_elem.text:
                    sql_query = command_text_elem.text.strip()
                    logger.info(f"Found SQL query for {dataset_name}: {sql_query[:100]}...")
                    
                    # Extract parameters used in this query
                    query_params = []
                    
                    # Try multiple methods for parameters too
                    param_elements = dataset.findall('.//rdl:QueryParameter', self.namespace_map)
                    if not param_elements:
                        param_elements = dataset.findall('.//QueryParameter')
                    if not param_elements and '}' in root.tag:
                        root_ns = root.tag.split('}')[0] + '}'
                        temp_ns_map = {'ns': root_ns.strip('{}')}
                        param_elements = dataset.findall('.//ns:QueryParameter', temp_ns_map)
                    
                    for param in param_elements:
                        param_name = param.get('Name', 'Unknown')
                        query_params.append(param_name)
                    
                    sql_queries.append({
                        'dataset_name': dataset_name,
                        'sql_query': sql_query,
                        'parameters': query_params,
                        'query_type': self._classify_query_type(sql_query)
                    })
                else:
                    logger.warning(f"No CommandText found for DataSet: {dataset_name}")
            
            logger.info(f"Successfully extracted {len(sql_queries)} SQL queries")
            return sql_queries
            
        except Exception as e:
            logger.error(f"Error extracting SQL queries: {str(e)}")
            return []
    
    def _classify_query_type(self, sql_query: str) -> str:
        """Classify the type of SQL query for better organization"""
        sql_upper = sql_query.upper()
        
        if 'SUM(' in sql_upper or 'COUNT(' in sql_upper or 'AVG(' in sql_upper:
            return 'Aggregation'
        elif 'JOIN' in sql_upper:
            return 'Join Query'
        elif 'WHERE' in sql_upper and any(op in sql_upper for op in ['BETWEEN', '>=', '<=', '>']):
            return 'Filtered Query'
        elif 'ORDER BY' in sql_upper:
            return 'Sorted Query'
        elif 'GROUP BY' in sql_upper:
            return 'Grouped Query'
        else:
            return 'Basic Query'
    
    def calculate_business_similarity(self, rdl1_path: str, rdl2_path: str) -> Dict[str, float]:
        """Calculate similarity based on business logic, not text matching"""
        
        context1 = self.analyze_rdl_file(rdl1_path)
        context2 = self.analyze_rdl_file(rdl2_path)
        
        # Business-focused similarity calculations
        similarities = {
            'data_source_similarity': self._compare_data_sources(context1, context2),
            'filter_logic_similarity': self._compare_business_filters(context1, context2),
            'calculation_similarity': self._compare_calculations(context1, context2),
            'output_purpose_similarity': self._compare_output_purpose(context1, context2),
            'parameter_similarity': self._compare_parameters(context1, context2),
            'business_purpose_similarity': self._compare_business_purpose(context1, context2)
        }
        
        # Calculate weighted overall similarity
        weights = {
            'business_purpose_similarity': 0.25,  # Most important
            'filter_logic_similarity': 0.25,     # Business rules
            'output_purpose_similarity': 0.20,   # What it produces
            'calculation_similarity': 0.15,      # How it calculates
            'data_source_similarity': 0.10,      # Where data comes from
            'parameter_similarity': 0.05         # User inputs
        }
        
        overall_similarity = sum(
            similarities[key] * weights[key] 
            for key in weights
        )
        
        similarities['overall_similarity'] = overall_similarity
        similarities['confidence_score'] = self._calculate_confidence(similarities)
        similarities['recommendation'] = self._generate_recommendation(similarities, context1, context2)
        
        return similarities
    
    def _calculate_business_similarity_from_contexts(self, context1: BusinessContext, context2: BusinessContext) -> Dict[str, float]:
        """Calculate similarity based on business logic using existing BusinessContext objects"""
        
        # Business-focused similarity calculations
        similarities = {
            'data_source_similarity': self._compare_data_sources(context1, context2),
            'filter_logic_similarity': self._compare_business_filters(context1, context2),
            'calculation_similarity': self._compare_calculations(context1, context2),
            'output_purpose_similarity': self._compare_output_purpose(context1, context2),
            'parameter_similarity': self._compare_parameters(context1, context2),
            'business_purpose_similarity': self._compare_business_purpose(context1, context2)
        }
        
        # Calculate weighted overall similarity - prioritize business domain separation
        weights = {
            'data_source_similarity': 0.30,      # important but not dominant
            'business_purpose_similarity': 0.35, # INCREASED - different business purposes should separate reports
            'filter_logic_similarity': 0.15,     # reduced
            'output_purpose_similarity': 0.15,   # increased - different outputs indicate different purposes
            'calculation_similarity': 0.03,      # reduced
            'parameter_similarity': 0.02         # reduced
        }
        
        overall_similarity = sum(
            similarities[key] * weights[key] 
            for key in weights
        )
        
        similarities['overall_similarity'] = overall_similarity
        similarities['confidence_score'] = self._calculate_confidence(similarities)
        similarities['recommendation'] = self._generate_recommendation(similarities, context1, context2)
        
        return similarities
    
    def _extract_data_sources(self, root: ET.Element) -> List[str]:
        """Extract data source information"""
        data_sources = []
        
        for ds in root.findall('.//rdl:DataSource', self.namespace_map):
            name = ds.get('Name', 'Unknown')
            
            # Get connection string to identify database/server
            conn_elem = ds.find('.//rdl:ConnectString', self.namespace_map)
            if conn_elem is not None:
                conn_str = conn_elem.text or ''
                # Extract meaningful identifiers from connection string
                db_name = self._extract_database_name(conn_str)
                data_sources.append(f"{name}:{db_name}")
            else:
                data_sources.append(name)
                
        return data_sources
    
    def _extract_business_filters(self, root: ET.Element) -> List[BusinessFilter]:
        """Extract business filters from SQL queries and parameters"""
        filters = []
        
        # Extract from SQL queries
        for dataset in root.findall('.//rdl:DataSet', self.namespace_map):
            query_elem = dataset.find('.//rdl:CommandText', self.namespace_map)
            if query_elem is not None and query_elem.text:
                sql_filters = self._parse_sql_filters(query_elem.text)
                filters.extend(sql_filters)
        
        # Extract from report parameters (user-driven filters)
        for param in root.findall('.//rdl:ReportParameter', self.namespace_map):
            param_filter = self._convert_parameter_to_filter(param)
            if param_filter:
                filters.append(param_filter)
        
        return filters
    
    def _parse_sql_filters(self, sql: str) -> List[BusinessFilter]:
        """Parse SQL WHERE clause to extract business logic filters"""
        filters = []
        
        try:
            parsed = sqlparse.parse(sql)[0]
            where_clause = self._find_where_clause(parsed)
            
            if where_clause:
                filter_conditions = self._extract_filter_conditions(where_clause)
                for condition in filter_conditions:
                    business_filter = self._classify_filter(condition)
                    if business_filter:
                        filters.append(business_filter)
                        
        except Exception as e:
            logger.warning(f"Could not parse SQL filters: {str(e)}")
            
        return filters
    
    def _extract_calculations(self, root: ET.Element) -> List[BusinessCalculation]:
        """Extract calculations, aggregations, and formulas"""
        calculations = []
        
        # Look for calculated fields and expressions
        for textbox in root.findall('.//rdl:Textbox', self.namespace_map):
            value_elem = textbox.find('.//rdl:Value', self.namespace_map)
            if value_elem is not None and value_elem.text:
                expression = value_elem.text
                
                # Check if this is a calculation
                if self._is_calculation_expression(expression):
                    calc = self._parse_calculation(textbox.get('Name', ''), expression)
                    if calc:
                        calculations.append(calc)
        
        return calculations
    
    def _extract_output_structure(self, root: ET.Element) -> BusinessOutput:
        """Analyze what the report actually outputs"""
        
        # Determine report type
        report_type = self._determine_report_type(root)
        
        # Extract output fields
        fields = self._extract_output_fields(root)
        
        # Extract grouping and sorting
        grouping_fields = self._extract_grouping_fields(root)
        sorting_fields = self._extract_sorting_fields(root)
        
        # Determine display format
        display_format = self._determine_display_format(root)
        
        return BusinessOutput(
            fields=fields,
            grouping_fields=grouping_fields,
            sorting_fields=sorting_fields,
            display_format=display_format,
            report_type=report_type
        )
    
    def _extract_parameters(self, root: ET.Element) -> Dict[str, Any]:
        """Extract report parameters and their business meaning"""
        parameters = {}
        
        for param in root.findall('.//rdl:ReportParameter', self.namespace_map):
            param_name = param.get('Name', '')
            param_info = {
                'data_type': param.get('DataType', 'String'),
                'allow_blank': param.get('AllowBlank', 'false') == 'true',
                'multivalue': param.get('MultiValue', 'false') == 'true',
                'business_purpose': self._infer_parameter_purpose(param_name)
            }
            
            # Extract default values
            default_elem = param.find('.//rdl:DefaultValue', self.namespace_map)
            if default_elem is not None:
                param_info['default_values'] = [
                    val.text for val in default_elem.findall('.//rdl:Value', self.namespace_map)
                    if val.text
                ]
            
            parameters[param_name] = param_info
            
        return parameters
    
    def _compare_business_filters(self, ctx1: BusinessContext, ctx2: BusinessContext) -> float:
        """Compare business filter logic (most critical for similarity)"""
        
        if not ctx1.filters and not ctx2.filters:
            return 100.0  # Both have no filters
        
        if not ctx1.filters or not ctx2.filters:
            return 0.0    # One filtered, one not = completely different purpose
        
        # Normalize and compare filter logic
        normalized_filters1 = self._normalize_filters(ctx1.filters)
        normalized_filters2 = self._normalize_filters(ctx2.filters)
        
        return self._calculate_filter_similarity(normalized_filters1, normalized_filters2)
    
    def _compare_data_sources(self, ctx1: BusinessContext, ctx2: BusinessContext) -> float:
        """Compare data sources and table domain compatibility"""
        # basic database/connection similarity
        db_similarity = self._compare_lists(ctx1.data_sources, ctx2.data_sources)
        
        # extract table domain compatibility from SQL queries
        tables1 = []
        tables2 = []
        
        # get tables from sql queries
        for query_info in ctx1.sql_queries:
            if 'sql_query' in query_info:
                # simple extraction of tables from SQL
                query_text = query_info['sql_query'].upper()
                import re
                # look for FROM and JOIN patterns
                table_matches = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)', query_text)
                tables1.extend(table_matches)
        
        for query_info in ctx2.sql_queries:
            if 'sql_query' in query_info:
                query_text = query_info['sql_query'].upper()
                import re
                table_matches = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)', query_text)
                tables2.extend(table_matches)
        
        # calculate domain compatibility if we have tables
        if tables1 and tables2:
            # use ai analyzer for domain classification
            if hasattr(self, 'ai_analyzer') and self.ai_analyzer:
                domain_compat = self.ai_analyzer._calculate_domain_compatibility(tables1, tables2)
                # weight both database and domain compatibility
                return (db_similarity * 0.3) + (domain_compat * 100 * 0.7)
            else:
                # fallback to simple table overlap with domain analysis
                table_overlap = len(set(tables1) & set(tables2)) / len(set(tables1) | set(tables2)) if (tables1 or tables2) else 0
                
                # Basic domain detection for tables
                domain_compatible = self._are_tables_domain_compatible(tables1, tables2)
                if not domain_compatible:
                    # Heavily penalize different domains
                    return min(25.0, (db_similarity * 0.2) + (table_overlap * 100 * 0.2))
                
                return (db_similarity * 0.3) + (table_overlap * 100 * 0.7)
        
        # if no tables found, just use database similarity
        return db_similarity
    
    def _compare_calculations(self, ctx1: BusinessContext, ctx2: BusinessContext) -> float:
        """Compare calculations and aggregations"""
        if not ctx1.calculations and not ctx2.calculations:
            return 100.0
        
        if not ctx1.calculations or not ctx2.calculations:
            return 0.0
        
        # Compare calculation types and expressions
        calc_sigs1 = {f"{c.calculation_type.value}:{c.aggregation_level}" for c in ctx1.calculations}
        calc_sigs2 = {f"{c.calculation_type.value}:{c.aggregation_level}" for c in ctx2.calculations}
        
        intersection = len(calc_sigs1 & calc_sigs2)
        union = len(calc_sigs1 | calc_sigs2)
        
        return (intersection / union * 100) if union > 0 else 0.0
    
    def _compare_parameters(self, ctx1: BusinessContext, ctx2: BusinessContext) -> float:
        """Compare report parameters"""
        if not ctx1.parameters and not ctx2.parameters:
            return 100.0
        
        if not ctx1.parameters or not ctx2.parameters:
            return 0.0
        
        # Compare parameter purposes
        purposes1 = {p['business_purpose'] for p in ctx1.parameters.values()}
        purposes2 = {p['business_purpose'] for p in ctx2.parameters.values()}
        
        intersection = len(purposes1 & purposes2)
        union = len(purposes1 | purposes2)
        
        return (intersection / union * 100) if union > 0 else 0.0
    
    def _compare_business_purpose(self, ctx1: BusinessContext, ctx2: BusinessContext) -> float:
        """Compare inferred business purposes"""
        return 100.0 if ctx1.business_purpose == ctx2.business_purpose else 0.0
    
    def _convert_parameter_to_filter(self, param_elem: ET.Element) -> Optional[BusinessFilter]:
        """Convert report parameter to business filter"""
        param_name = param_elem.get('Name', '')
        if not param_name:
            return None
        
        # Infer filter type from parameter name and type
        data_type = param_elem.get('DataType', 'String')
        purpose = self._infer_parameter_purpose(param_name)
        
        filter_type_map = {
            'date_filter': FilterType.DATE_RANGE,
            'location_filter': FilterType.CATEGORICAL,
            'product_filter': FilterType.CATEGORICAL,
            'customer_filter': FilterType.CATEGORICAL,
            'employee_filter': FilterType.CATEGORICAL,
            'general_filter': FilterType.CATEGORICAL
        }
        
        filter_type = filter_type_map.get(purpose, FilterType.CATEGORICAL)
        
        return BusinessFilter(
            field_name=param_name,
            filter_type=filter_type,
            condition=f"Parameter-driven filter: {param_name}",
            values=[],
            is_parameter_driven=True,
            parameter_name=param_name
        )
    
    def _compare_output_purpose(self, ctx1: BusinessContext, ctx2: BusinessContext) -> float:
        """Compare what the reports actually produce"""
        
        similarity_scores = []
        
        # Compare field overlap (exact matches)
        fields1 = set(f.lower() for f in ctx1.output.fields)
        fields2 = set(f.lower() for f in ctx2.output.fields)
        
        if fields1 or fields2:
            union_size = len(fields1 | fields2)
            if union_size > 0:
                field_similarity = len(fields1 & fields2) / union_size * 100
                similarity_scores.append(field_similarity * 1.5)  # Weight exact field matches highly
        
        # Compare business domain based on field names
        domain_similarity = self._compare_business_domains(ctx1.output.fields, ctx2.output.fields)
        similarity_scores.append(domain_similarity)
        
        # Compare report types
        type_similarity = 100.0 if ctx1.output.report_type == ctx2.output.report_type else 0.0
        similarity_scores.append(type_similarity * 0.5)  # Lower weight for report type
        
        # Compare grouping logic
        group_similarity = self._compare_lists(ctx1.output.grouping_fields, ctx2.output.grouping_fields)
        similarity_scores.append(group_similarity)
        
        return min(100.0, sum(similarity_scores) / len(similarity_scores)) if similarity_scores else 0.0
    
    def _compare_business_domains(self, fields1: List[str], fields2: List[str]) -> float:
        """Compare business domains based on field semantics"""
        
        # Define business domain keywords - more comprehensive and specific
        domain_keywords = {
            'sales': ['sale', 'revenue', 'price', 'quantity', 'product', 'order', 'invoice', 'sales', 'sold', 'selling', 'purchase'],
            'customer': ['customer', 'client', 'contact', 'email', 'phone', 'address', 'name', 'customers'],
            'financial': ['amount', 'cost', 'profit', 'expense', 'budget', 'payment', 'financial', 'money', 'dollar', 'currency'],
            'inventory': ['stock', 'warehouse', 'item', 'category', 'supplier', 'inventory', 'reorder', 'stocking', 'supply'],
            'hr': ['employee', 'staff', 'department', 'salary', 'hire', 'position', 'human', 'resource', 'payroll'],
            'analytics': ['count', 'sum', 'average', 'total', 'percentage', 'ratio', 'analysis', 'metric', 'kpi'],
            'operations': ['process', 'operation', 'workflow', 'task', 'activity', 'production', 'manufacturing'],
            'marketing': ['campaign', 'lead', 'prospect', 'marketing', 'promotion', 'advertisement', 'channel']
        }
        
        # Determine domain for each field set
        domain1 = self._classify_field_domain(fields1, domain_keywords)
        domain2 = self._classify_field_domain(fields2, domain_keywords)
        
        # If same domain, high similarity; if different, very low similarity
        if domain1 == domain2 and domain1 != 'unknown':
            return 85.0
        elif domain1 != 'unknown' and domain2 != 'unknown' and domain1 != domain2:
            return 5.0   # VERY LOW - Different business domains should not be consolidated
        else:
            return 25.0  # Unknown domain - conservative approach
    
    def _classify_field_domain(self, fields: List[str], domain_keywords: Dict[str, List[str]]) -> str:
        """Classify fields into business domain"""
        field_text = ' '.join(fields).lower()
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in field_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'unknown'
    
    def _are_tables_domain_compatible(self, tables1: List[str], tables2: List[str]) -> bool:
        """Check if table sets belong to compatible business domains"""
        # Define table name patterns for different domains
        domain_patterns = {
            'sales': ['sales', 'order', 'invoice', 'product', 'customer'],
            'inventory': ['inventory', 'stock', 'warehouse', 'item', 'supplier'],
            'hr': ['employee', 'staff', 'department', 'payroll', 'hire'],
            'financial': ['finance', 'accounting', 'budget', 'expense', 'payment'],
            'operations': ['production', 'manufacturing', 'operation', 'process']
        }
        
        def classify_tables(tables):
            table_text = ' '.join([t.lower() for t in tables])
            scores = {}
            for domain, patterns in domain_patterns.items():
                score = sum(1 for pattern in patterns if pattern in table_text)
                if score > 0:
                    scores[domain] = score
            return max(scores, key=scores.get) if scores else 'unknown'
        
        domain1 = classify_tables(tables1)
        domain2 = classify_tables(tables2)
        
        # Same domain or one is unknown = compatible
        return domain1 == domain2 or domain1 == 'unknown' or domain2 == 'unknown'
    
    def _infer_business_purpose(self, filters: List[BusinessFilter], 
                              calculations: List[BusinessCalculation], 
                              output: BusinessOutput) -> str:
        """Infer the business purpose of the report from its structure"""
        
        # Analyze patterns to determine purpose
        has_date_filters = any(f.filter_type == FilterType.DATE_RANGE for f in filters)
        has_aggregations = any(c.calculation_type == CalculationType.AGGREGATION for c in calculations)
        is_detailed = len(output.fields) > 10
        is_summary = len(output.grouping_fields) > 0
        
        if has_date_filters and has_aggregations and is_summary:
            return "periodic_summary_report"
        elif has_date_filters and is_detailed:
            return "detailed_transactional_report"
        elif has_aggregations and is_summary:
            return "analytical_summary_report"
        elif is_detailed and not filters:
            return "master_data_report"
        else:
            return "custom_operational_report"
    
    # Helper methods (implementation details)
    def _extract_database_name(self, conn_str: str) -> str:
        """Extract database name from connection string"""
        match = re.search(r'Initial Catalog=([^;]+)', conn_str, re.IGNORECASE)
        return match.group(1) if match else 'unknown_db'
    
    def _find_where_clause(self, parsed_sql):
        """Find WHERE clause in parsed SQL"""
        return RDLParserHelpers.find_where_clause(parsed_sql)
    
    def _extract_filter_conditions(self, where_clause):
        """Extract filter conditions from WHERE clause"""
        return RDLParserHelpers.extract_filter_conditions(where_clause)
    
    def _classify_filter(self, condition):
        """Classify a filter condition"""
        return RDLParserHelpers.classify_filter(condition)
    
    def _is_calculation_expression(self, expression: str) -> bool:
        """Check if expression contains calculations"""
        calc_patterns = [
            r'=Sum\(',
            r'=Count\(',
            r'=Avg\(',
            r'=Max\(',
            r'=Min\(',
            r'=.*\+.*',
            r'=.*\-.*',
            r'=.*\*.*',
            r'=.*/.*'
        ]
        return any(re.search(pattern, expression, re.IGNORECASE) for pattern in calc_patterns)
    
    def _parse_calculation(self, name: str, expression: str):
        """Parse calculation expression"""
        return RDLParserHelpers.parse_calculation(name, expression)
    
    def _determine_report_type(self, root: ET.Element) -> str:
        """Determine report type"""
        return RDLParserHelpers.determine_report_type(root)
    
    def _extract_output_fields(self, root: ET.Element) -> List[str]:
        """Extract output fields"""
        return RDLParserHelpers.extract_output_fields(root)
    
    def _extract_grouping_fields(self, root: ET.Element) -> List[str]:
        """Extract grouping fields"""
        return RDLParserHelpers.extract_grouping_fields(root)
    
    def _extract_sorting_fields(self, root: ET.Element) -> List[str]:
        """Extract sorting fields"""
        return RDLParserHelpers.extract_sorting_fields(root)
    
    def _determine_display_format(self, root: ET.Element) -> str:
        """Determine display format"""
        # Simple implementation - could be enhanced
        return "standard_table"
    
    def _infer_parameter_purpose(self, param_name: str) -> str:
        """Infer parameter purpose"""
        return RDLParserHelpers.infer_parameter_purpose(param_name)
    
    def _normalize_filters(self, filters: List[BusinessFilter]) -> List[Dict[str, Any]]:
        """Normalize filters for comparison"""
        return RDLParserHelpers.normalize_filters(filters)
    
    def _calculate_filter_similarity(self, filters1: List[Dict[str, Any]], 
                                   filters2: List[Dict[str, Any]]) -> float:
        """Calculate filter similarity"""
        return RDLParserHelpers.calculate_filter_similarity(filters1, filters2)
    
    def _compare_lists(self, list1: List[str], list2: List[str]) -> float:
        """Compare two lists"""
        return RDLParserHelpers.compare_lists(list1, list2)
    
    def _calculate_confidence(self, similarities: Dict[str, float]) -> float:
        """Calculate confidence score for the similarity assessment"""
        # Higher confidence when multiple dimensions agree
        high_similarities = [s for s in similarities.values() if isinstance(s, (int, float)) and s > 80]
        low_similarities = [s for s in similarities.values() if isinstance(s, (int, float)) and s < 20]
        
        if len(high_similarities) >= 3:
            return 95.0
        elif len(low_similarities) >= 3:
            return 90.0
        else:
            return 70.0
    
    def _generate_recommendation(self, similarities: Dict[str, float], 
                                ctx1: BusinessContext, ctx2: BusinessContext) -> str:
        """Generate human-readable recommendation"""
        overall = similarities.get('overall_similarity', 0)
        
        if overall > 85:
            return f"HIGH SIMILARITY: These reports serve similar business purposes. Consider consolidation."
        elif overall > 50:
            return f"MODERATE SIMILARITY: Some overlap detected, review business logic for potential consolidation."
        else:
            return f"LOW SIMILARITY: Reports serve different business purposes, migrate separately."
    
    def analyze_rdl_batch_ai_enhanced(self, rdl_files: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        AI-enhanced batch analysis of RDL files with cost optimization.
        
        Args:
            rdl_files: List of dicts with 'path', 'content', 'filename' keys
            
        Returns:
            Comprehensive analysis results with similarity pairs and recommendations
        """
        logger.info(f"Starting AI-enhanced batch analysis of {len(rdl_files)} RDL files")
        
        # Step 1: Extract SQL queries and basic context from all RDL files
        query_data = {}
        business_contexts = {}
        file_sql_queries = {}  # Track SQL queries per file for frontend
        
        for i, rdl_file in enumerate(rdl_files):
            try:
                file_id = f"file_{i}_{rdl_file['filename']}"
                
                # Check cache first
                cache_key = f"rdl_analysis_{hashlib.md5(rdl_file['content'].encode()).hexdigest()}"
                if self.cache_manager:
                    cached_result = self.cache_manager.get(cache_key)
                    if cached_result:
                        business_contexts[file_id] = cached_result['context']
                        query_data.update(cached_result['queries'])
                        file_sql_queries[rdl_file['filename']] = cached_result.get('file_queries', [])
                        self.analysis_stats['cache_hits'] += 1
                        continue
                
                # Analyze RDL file
                context = self.analyze_rdl_file(rdl_file['path'])
                business_contexts[file_id] = context
                
                # Enhanced query extraction with more context
                context_sql_queries = getattr(context, 'sql_queries', [])
                file_sql_queries[rdl_file['filename']] = context_sql_queries
                
                # Extract queries for AI analysis
                if context_sql_queries:
                    for j, query_info in enumerate(context_sql_queries):
                        query_id = f"{file_id}_query_{j}"
                        query_data[query_id] = {
                            'id': query_id,
                            'query': query_info['sql_query'],
                            'source_file': rdl_file['filename'],
                            'dataset_name': query_info.get('dataset_name', 'unknown'),
                            'query_type': query_info.get('query_type', 'Basic Query'),
                            'full_query_details': query_info  # Include full query details
                        }
                
                # Cache the result with full query information
                if self.cache_manager:
                    cache_data = {
                        'context': context,
                        'queries': {qid: qdata for qid, qdata in query_data.items() 
                                  if qdata['source_file'] == rdl_file['filename']},
                        'file_queries': context_sql_queries
                    }
                    self.cache_manager.put(cache_key, cache_data, ttl_hours=24, 
                                         tags=['rdl_analysis', rdl_file['filename']])
                
            except Exception as e:
                logger.error(f"Error analyzing RDL file {rdl_file['filename']}: {e}")
                continue
        
        logger.info(f"Extracted {len(query_data)} SQL queries from {len(business_contexts)} RDL files")
        self.analysis_stats['total_queries_analyzed'] = len(query_data)
        
        # Step 2: AI-powered query similarity analysis
        similarity_results = []
        if self.enable_ai_analysis and query_data:
            similarity_results = self._perform_ai_similarity_analysis(query_data)
        
        # Enrich similarity results with actual SQL queries
        for result in similarity_results:
            q1_info = query_data.get(result['query1_id'], {})
            q2_info = query_data.get(result['query2_id'], {})
            
            # Get SQL queries directly from query_data since file_sql_queries might be empty
            file1_name = q1_info.get('source_file', '')
            file2_name = q2_info.get('source_file', '')
            
            # Find all queries for file1 and file2 from query_data
            file1_queries = []
            file2_queries = []
            
            for query_id, query_info in query_data.items():
                if query_info.get('source_file') == file1_name:
                    file1_queries.append({
                        'dataset_name': query_info.get('dataset_name', ''),
                        'sql_query': query_info.get('query', ''),
                        'parameters': [],  # Parameters would need to be extracted separately
                        'query_type': query_info.get('query_type', '')
                    })
                elif query_info.get('source_file') == file2_name:
                    file2_queries.append({
                        'dataset_name': query_info.get('dataset_name', ''),
                        'sql_query': query_info.get('query', ''),
                        'parameters': [],
                        'query_type': query_info.get('query_type', '')
                    })
            
            result['sql_queries'] = {
                'file1_queries': file1_queries,
                'file2_queries': file2_queries
            }
        
        # Step 3: Traditional business logic analysis for comparison
        traditional_pairs = self._perform_traditional_analysis(business_contexts)
        
        # Step 4: Combine and rank results
        combined_results = self._combine_analysis_results(
            similarity_results, traditional_pairs, business_contexts, query_data
        )
        
        # Step 5: Generate consolidation recommendations
        recommendations = self._generate_batch_recommendations(
            combined_results, business_contexts, query_data
        )
        
        logger.info(f"Analysis complete. Found {len(combined_results)} similar pairs")
        
        return {
            'total_files': len(rdl_files),
            'total_queries': len(query_data),
            'similarity_pairs': combined_results,
            'consolidation_recommendations': recommendations,
            'analysis_stats': self._get_enhanced_stats(),
            'query_details': query_data,
            'file_sql_queries': file_sql_queries  # Added for frontend query preview
        }
    
    def _perform_ai_similarity_analysis(self, query_data: Dict[str, Dict]) -> List[Dict]:
        """Perform AI-powered similarity analysis with cost optimization"""
        if not self.ai_analyzer:
            return []
        
        logger.info("Performing AI-powered query analysis...")
        
        # Preprocess queries for better analysis
        if self.query_preprocessor:
            preprocessed_queries = self.query_preprocessor.batch_preprocess(
                {qid: qdata['query'] for qid, qdata in query_data.items()}
            )
        else:
            preprocessed_queries = {}
        
        # Convert to format expected by AI analyzer
        ai_queries = []
        for query_id, query_info in query_data.items():
            ai_queries.append({
                'id': query_id,
                'query': query_info['query'],
                'source_file': query_info['source_file']
            })
        
        # Perform AI analysis
        query_analyses = self.ai_analyzer.analyze_query_batch(ai_queries)
        similarity_pairs = self.ai_analyzer.find_similar_queries(
            query_analyses, similarity_threshold=0.4
        )
        
        # Convert AI results to our format
        ai_results = []
        for sim_pair in similarity_pairs:
            q1_info = query_data.get(sim_pair.query1_id, {})
            q2_info = query_data.get(sim_pair.query2_id, {})
            
            ai_results.append({
                'file1': q1_info.get('source_file', 'unknown'),
                'file2': q2_info.get('source_file', 'unknown'),
                'similarity': sim_pair.overall_similarity * 100,
                'confidence': sim_pair.confidence_score * 100,
                'analysis_type': 'ai_powered',
                'details': {
                    'query_semantics': sim_pair.dimension_scores.get('query_semantics', 0) * 100,
                    'data_sources': sim_pair.dimension_scores.get('data_sources', 0) * 100,
                    'business_logic': sim_pair.dimension_scores.get('business_logic', 0) * 100,
                    'output_structure': sim_pair.dimension_scores.get('output_structure', 0) * 100,
                    'parameters': sim_pair.dimension_scores.get('parameters', 0) * 100
                },
                'ai_explanation': sim_pair.ai_explanation,
                'recommendation': sim_pair.consolidation_recommendation,
                'token_cost': sim_pair.token_cost,
                'query1_id': sim_pair.query1_id,
                'query2_id': sim_pair.query2_id
            })
        
        self.analysis_stats['ai_analyses_performed'] = len(ai_results)
        self.analysis_stats['token_usage'] = sum(r['token_cost'] for r in ai_results)
        
        logger.info(f"AI analysis complete. Found {len(ai_results)} similar query pairs")
        return ai_results
    
    def _perform_traditional_analysis(self, business_contexts: Dict[str, BusinessContext]) -> List[Dict]:
        """Perform traditional business logic analysis"""
        traditional_pairs = []
        
        context_items = list(business_contexts.items())
        for i in range(len(context_items)):
            for j in range(i + 1, len(context_items)):
                file1_id, context1 = context_items[i]
                file2_id, context2 = context_items[j]
                
                # Use business similarity calculation with BusinessContext objects
                similarities = self._calculate_business_similarity_from_contexts(
                    context1, context2
                )
                
                if similarities['overall_similarity'] > 30:  # Only include reasonable matches
                    traditional_pairs.append({
                        'file1': file1_id,
                        'file2': file2_id,
                        'similarity': similarities['overall_similarity'],
                        'confidence': similarities['confidence_score'],
                        'analysis_type': 'traditional',
                        'details': {
                            'business_purpose_similarity': similarities['business_purpose_similarity'],
                            'filter_logic_similarity': similarities['filter_logic_similarity'],
                            'output_purpose_similarity': similarities['output_purpose_similarity'],
                            'calculation_similarity': similarities['calculation_similarity'],
                            'data_source_similarity': similarities['data_source_similarity']
                        },
                        'recommendation': similarities['recommendation']
                    })
        
        return traditional_pairs
    
    def _combine_analysis_results(self, ai_results: List[Dict], 
                                traditional_results: List[Dict],
                                business_contexts: Dict[str, BusinessContext],
                                query_data: Dict[str, Dict] = None) -> List[Dict]:
        """Combine AI and traditional analysis results"""
        combined = []
        
        # Create lookup for combining results
        ai_lookup = {}
        for result in ai_results:
            key = tuple(sorted([result['file1'], result['file2']]))
            ai_lookup[key] = result
        
        traditional_lookup = {}
        for result in traditional_results:
            key = tuple(sorted([result['file1'], result['file2']]))
            traditional_lookup[key] = result
        
        # Get all unique file pairs
        all_pairs = set(ai_lookup.keys()) | set(traditional_lookup.keys())
        
        for pair in all_pairs:
            file1, file2 = pair
            ai_result = ai_lookup.get(pair)
            traditional_result = traditional_lookup.get(pair)
            
            if ai_result and traditional_result:
                # Combine both analyses with weighted average
                combined_similarity = (ai_result['similarity'] * 0.7 + 
                                     traditional_result['similarity'] * 0.3)
                combined_confidence = (ai_result['confidence'] * 0.7 + 
                                     traditional_result['confidence'] * 0.3)
                
                combined_result = {
                    'file1': file1,
                    'file2': file2,
                    'similarity': combined_similarity,
                    'confidence': combined_confidence,
                    'analysis_type': 'combined',
                    'ai_details': ai_result.get('details', {}),
                    'traditional_details': traditional_result.get('details', {}),
                    'ai_explanation': ai_result.get('ai_explanation', ''),
                    'recommendation': self._determine_combined_recommendation(combined_similarity),
                    'token_cost': ai_result.get('token_cost', 0),
                    'query1_id': ai_result.get('query1_id'),
                    'query2_id': ai_result.get('query2_id')
                }
                combined.append(combined_result)
                
            elif ai_result:
                combined.append(ai_result)
            elif traditional_result:
                # Enrich traditional result with SQL queries if query_data is available
                if query_data:
                    # Find SQL queries for this traditional pair
                    file1_queries = []
                    file2_queries = []
                    
                    # Normalize filenames - remove file_X_ prefix if present
                    def normalize_filename(filename):
                        import re
                        return re.sub(r'^file_\d+_', '', filename)
                    
                    normalized_file1 = normalize_filename(file1)
                    normalized_file2 = normalize_filename(file2)
                    
                    for query_id, query_info in query_data.items():
                        source_file = query_info.get('source_file', '')
                        if source_file == normalized_file1:
                            file1_queries.append({
                                'dataset_name': query_info.get('dataset_name', ''),
                                'sql_query': query_info.get('query', ''),
                                'parameters': [],
                                'query_type': query_info.get('query_type', '')
                            })
                        elif source_file == normalized_file2:
                            file2_queries.append({
                                'dataset_name': query_info.get('dataset_name', ''),
                                'sql_query': query_info.get('query', ''),
                                'parameters': [],
                                'query_type': query_info.get('query_type', '')
                            })
                    
                    # Add SQL queries to traditional result
                    traditional_result['sql_queries'] = {
                        'file1_queries': file1_queries,
                        'file2_queries': file2_queries
                    }
                
                combined.append(traditional_result)
        
        # Sort by similarity score
        combined.sort(key=lambda x: x['similarity'], reverse=True)
        return combined
    
    def _determine_combined_recommendation(self, similarity: float) -> str:
        """Determine recommendation based on combined similarity score"""
        if similarity >= 80:
            return "HIGH_SIMILARITY"
        elif similarity >= 50:
            return "MEDIUM_SIMILARITY"
        else:
            return "LOW_SIMILARITY"
    
    def _generate_batch_recommendations(self, similarity_pairs: List[Dict],
                                       business_contexts: Dict[str, BusinessContext],
                                       query_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive consolidation recommendations"""
        
        high_similarity_pairs = [p for p in similarity_pairs if p['similarity'] >= 80]
        medium_similarity_pairs = [p for p in similarity_pairs if 50 <= p['similarity'] < 80]
        
        # Group by potential consolidation clusters
        consolidation_clusters = self._identify_consolidation_clusters(high_similarity_pairs)
        
        return {
            'total_pairs_analyzed': len(similarity_pairs),
            'high_similarity_count': len(high_similarity_pairs),
            'medium_similarity_count': len(medium_similarity_pairs),
            'consolidation_clusters': consolidation_clusters,
            'estimated_cost_savings': self._calculate_cost_savings(consolidation_clusters),
            'migration_priority': self._suggest_migration_priority(similarity_pairs),
            'token_usage_summary': {
                'total_tokens': self.analysis_stats.get('token_usage', 0),
                'estimated_cost': self.analysis_stats.get('token_usage', 0) * 0.002 / 1000,
                'queries_analyzed': self.analysis_stats.get('total_queries_analyzed', 0)
            }
        }
    
    def _identify_consolidation_clusters(self, high_similarity_pairs: List[Dict]) -> List[Dict]:
        """Identify clusters of files that can be consolidated together"""
        clusters = []
        processed_files = set()
        
        for pair in high_similarity_pairs:
            file1, file2 = pair['file1'], pair['file2']
            
            if file1 not in processed_files and file2 not in processed_files:
                # Start a new cluster
                cluster_files = {file1, file2}
                cluster_similarities = [pair['similarity']]
                
                # Find other files that are similar to any in this cluster
                for other_pair in high_similarity_pairs:
                    if other_pair != pair:
                        other_file1, other_file2 = other_pair['file1'], other_pair['file2']
                        if (other_file1 in cluster_files and other_file2 not in processed_files):
                            cluster_files.add(other_file2)
                            cluster_similarities.append(other_pair['similarity'])
                        elif (other_file2 in cluster_files and other_file1 not in processed_files):
                            cluster_files.add(other_file1)
                            cluster_similarities.append(other_pair['similarity'])
                
                if len(cluster_files) >= 2:
                    clusters.append({
                        'files': list(cluster_files),
                        'average_similarity': sum(cluster_similarities) / len(cluster_similarities) if cluster_similarities else 0,
                        'cluster_size': len(cluster_files),
                        'consolidation_potential': 'HIGH' if len(cluster_files) >= 3 else 'MEDIUM'
                    })
                    
                    processed_files.update(cluster_files)
        
        return clusters
    
    def _calculate_cost_savings(self, clusters: List[Dict]) -> Dict[str, float]:
        """Calculate potential cost savings from consolidation"""
        total_files = sum(cluster['cluster_size'] for cluster in clusters)
        consolidated_files = len(clusters)
        
        # Rough estimates based on typical migration effort
        original_effort_hours = total_files * 8  # 8 hours per report
        consolidated_effort_hours = consolidated_files * 12  # 12 hours per consolidated report
        
        effort_savings = max(0, original_effort_hours - consolidated_effort_hours)
        cost_savings = effort_savings * 75  # $75 per hour estimate
        
        return {
            'effort_hours_saved': effort_savings,
            'estimated_cost_savings': cost_savings,
            'consolidation_ratio': (total_files - consolidated_files) / total_files if total_files > 0 else 0
        }
    
    def _suggest_migration_priority(self, similarity_pairs: List[Dict]) -> List[Dict]:
        """Suggest migration priority based on similarity analysis"""
        # Group files by involvement in high-similarity pairs
        file_similarity_scores = {}
        
        for pair in similarity_pairs:
            file1, file2 = pair['file1'], pair['file2']
            score = pair['similarity']
            
            file_similarity_scores[file1] = file_similarity_scores.get(file1, [])
            file_similarity_scores[file1].append(score)
            
            file_similarity_scores[file2] = file_similarity_scores.get(file2, [])
            file_similarity_scores[file2].append(score)
        
        # Calculate average similarity for each file
        file_priorities = []
        for filename, scores in file_similarity_scores.items():
            avg_similarity = sum(scores) / len(scores) if scores else 0
            file_priorities.append({
                'filename': filename,
                'average_similarity': avg_similarity,
                'similar_file_count': len(scores),
                'priority': 'HIGH' if avg_similarity >= 75 and len(scores) >= 3 else 
                          'MEDIUM' if avg_similarity >= 50 else 'LOW'
            })
        
        # Sort by consolidation potential
        file_priorities.sort(key=lambda x: (x['similar_file_count'], x['average_similarity']), reverse=True)
        
        return file_priorities
    
    def _get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced analysis statistics"""
        ai_stats = {}
        if self.ai_analyzer:
            ai_stats = self.ai_analyzer.get_analysis_stats()
        
        embedding_stats = {}
        if self.embedding_manager:
            embedding_stats = self.embedding_manager.get_stats()
        
        cache_stats = {}
        if self.cache_manager:
            cache_stats = self.cache_manager.get_cache_stats()
        
        return {
            **self.analysis_stats,
            'ai_analyzer_stats': ai_stats,
            'embedding_stats': embedding_stats,
            'cache_stats': cache_stats
        }

# Usage example
if __name__ == "__main__":
    analyzer = RDLBusinessAnalyzer()
    
    # Test with our sample files
    similarity = analyzer.calculate_business_similarity(
        "/home/reddysandeep/Documents/ssrs to powerbi/SalesReport.rdl",
        "/home/reddysandeep/Documents/ssrs to powerbi/SalesReport_Q1.rdl"
    )
    
    print("Business Logic Similarity Analysis:")
    print("=" * 50)
    for key, value in similarity.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value}")