"""
RDL Business Logic Analyzer
==========================
Enterprise-grade similarity analysis based on business logic, not text similarity.
Designed specifically for SSRS RDL reports to identify true functional duplicates.
"""

import xml.etree.ElementTree as ET
import re
import sqlparse
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import logging
from rdl_types import (
    FilterType, CalculationType, BusinessFilter, BusinessCalculation, 
    BusinessOutput, BusinessContext
)
from rdl_parser_helpers import RDLParserHelpers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RDLBusinessAnalyzer:
    """
    Analyzes RDL files to extract business logic and calculate meaningful similarity.
    Unlike generic text similarity, this understands SSRS reporting patterns.
    """
    
    def __init__(self):
        self.namespace_map = {
            'rdl': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition',
            'rd': 'http://schemas.microsoft.com/SQLServer/reporting/reportdesigner'
        }
        # Will be set dynamically based on actual file namespace
    
    def _detect_namespace(self, root):
        """Detect the correct RDL namespace from the root element"""
        root_tag = root.tag
        if '}' in root_tag:
            namespace = root_tag.split('}')[0] + '}'
            # Update namespace mapping with detected namespace
            self.namespace_map['rdl'] = namespace.strip('{}')
            logger.debug(f"Detected RDL namespace: {self.namespace_map['rdl']}")
        
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
            # Find all DataSet elements
            datasets = root.findall('.//rdl:DataSet', self.namespace_map)
            
            for dataset in datasets:
                dataset_name = dataset.get('Name', 'Unknown')
                
                # Find CommandText within Query
                command_text_elem = dataset.find('.//rdl:CommandText', self.namespace_map)
                if command_text_elem is not None and command_text_elem.text:
                    sql_query = command_text_elem.text.strip()
                    
                    # Extract parameters used in this query
                    query_params = []
                    param_elements = dataset.findall('.//rdl:QueryParameter', self.namespace_map)
                    for param in param_elements:
                        param_name = param.get('Name', 'Unknown')
                        query_params.append(param_name)
                    
                    sql_queries.append({
                        'dataset_name': dataset_name,
                        'sql_query': sql_query,
                        'parameters': query_params,
                        'query_type': self._classify_query_type(sql_query)
                    })
            
            return sql_queries
            
        except Exception as e:
            logger.warning(f"Error extracting SQL queries: {str(e)}")
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
        """Compare data sources"""
        return self._compare_lists(ctx1.data_sources, ctx2.data_sources)
    
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
            field_similarity = len(fields1 & fields2) / len(fields1 | fields2) * 100
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
        
        # Define business domain keywords
        domain_keywords = {
            'sales': ['sale', 'revenue', 'price', 'quantity', 'product', 'order', 'invoice'],
            'customer': ['customer', 'client', 'contact', 'email', 'phone', 'address', 'name'],
            'financial': ['amount', 'cost', 'profit', 'expense', 'budget', 'payment'],
            'inventory': ['stock', 'warehouse', 'item', 'category', 'supplier'],
            'hr': ['employee', 'staff', 'department', 'salary', 'hire', 'position'],
            'analytics': ['count', 'sum', 'average', 'total', 'percentage', 'ratio']
        }
        
        # Determine domain for each field set
        domain1 = self._classify_field_domain(fields1, domain_keywords)
        domain2 = self._classify_field_domain(fields2, domain_keywords)
        
        # If same domain, high similarity; if different, low similarity
        if domain1 == domain2 and domain1 != 'unknown':
            return 90.0
        elif domain1 != 'unknown' and domain2 != 'unknown' and domain1 != domain2:
            return 10.0  # Different business domains
        else:
            return 50.0  # Unknown domain
    
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