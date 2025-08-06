"""
RDL Parser Helper Functions
===========================
Detailed implementation of parsing functions for extracting business logic from RDL files.
"""

import xml.etree.ElementTree as ET
import re
import sqlparse
from sqlparse.sql import Statement, Token, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, Name, Operator, Number, String
from typing import Dict, List, Optional, Tuple, Any
from src.models.rdl_models import BusinessFilter, BusinessCalculation, FilterType, CalculationType

class RDLParserHelpers:
    """Helper functions for parsing RDL files and extracting business logic"""
    
    @staticmethod
    def find_where_clause(parsed_sql: Statement) -> Optional[Where]:
        """Find WHERE clause in parsed SQL statement"""
        for token in parsed_sql.tokens:
            if isinstance(token, Where):
                return token
            elif hasattr(token, 'tokens'):
                # Recursively search in nested tokens
                where_clause = RDLParserHelpers._find_where_recursive(token)
                if where_clause:
                    return where_clause
        return None
    
    @staticmethod
    def _find_where_recursive(token) -> Optional[Where]:
        """Recursively search for WHERE clause in token tree"""
        if isinstance(token, Where):
            return token
        if hasattr(token, 'tokens'):
            for sub_token in token.tokens:
                result = RDLParserHelpers._find_where_recursive(sub_token)
                if result:
                    return result
        return None
    
    @staticmethod
    def extract_filter_conditions(where_clause: Where) -> List[Dict[str, Any]]:
        """Extract individual filter conditions from WHERE clause"""
        conditions = []
        
        # Parse the WHERE clause tokens
        condition_text = str(where_clause).replace('WHERE', '').strip()
        
        # Split by AND/OR (simple approach - could be enhanced)
        and_conditions = re.split(r'\s+AND\s+', condition_text, flags=re.IGNORECASE)
        
        for condition in and_conditions:
            condition = condition.strip()
            if condition:
                parsed_condition = RDLParserHelpers._parse_single_condition(condition)
                if parsed_condition:
                    conditions.append(parsed_condition)
        
        return conditions
    
    @staticmethod
    def _parse_single_condition(condition: str) -> Optional[Dict[str, Any]]:
        """Parse a single filter condition"""
        condition = condition.strip()
        
        # Date range patterns
        date_patterns = [
            (r'(\w+)\s*=\s*@(\w+)', 'parameter_date'),
            (r'(\w+)\s*BETWEEN\s*(.+)\s*AND\s*(.+)', 'date_range'),
            (r'MONTH\((\w+)\)\s*IN\s*\(([^)]+)\)', 'month_filter'),
            (r'YEAR\((\w+)\)\s*=\s*(\d+)', 'year_filter'),
            (r'(\w+)\s*>=\s*[\'"]([^\'"]+)[\'"]', 'date_gte'),
            (r'(\w+)\s*<=\s*[\'"]([^\'"]+)[\'"]', 'date_lte')
        ]
        
        # Categorical patterns
        categorical_patterns = [
            (r'(\w+)\s*IN\s*\(([^)]+)\)', 'in_list'),
            (r'(\w+)\s*=\s*[\'"]([^\'"]+)[\'"]', 'equals_string'),
            (r'(\w+)\s*LIKE\s*[\'"]([^\'"]+)[\'"]', 'like_pattern')
        ]
        
        # Numeric patterns
        numeric_patterns = [
            (r'(\w+)\s*>\s*(\d+(?:\.\d+)?)', 'greater_than'),
            (r'(\w+)\s*<\s*(\d+(?:\.\d+)?)', 'less_than'),
            (r'(\w+)\s*=\s*(\d+(?:\.\d+)?)', 'equals_number')
        ]
        
        # Try to match patterns
        for patterns, category in [(date_patterns, 'date'), 
                                  (categorical_patterns, 'categorical'),
                                  (numeric_patterns, 'numeric')]:
            for pattern, filter_type in patterns:
                match = re.search(pattern, condition, re.IGNORECASE)
                if match:
                    return {
                        'field_name': match.group(1),
                        'filter_type': filter_type,
                        'category': category,
                        'condition': condition,
                        'values': list(match.groups()[1:]) if len(match.groups()) > 1 else []
                    }
        
        return None
    
    @staticmethod
    def classify_filter(condition_dict: Dict[str, Any]) -> Optional[BusinessFilter]:
        """Convert parsed condition to BusinessFilter object"""
        if not condition_dict:
            return None
        
        field_name = condition_dict['field_name']
        condition = condition_dict['condition']
        values = condition_dict['values']
        
        # Determine filter type based on pattern
        filter_type_map = {
            'month_filter': FilterType.DATE_RANGE,
            'year_filter': FilterType.DATE_RANGE,
            'date_range': FilterType.DATE_RANGE,
            'date_gte': FilterType.DATE_RANGE,
            'date_lte': FilterType.DATE_RANGE,
            'parameter_date': FilterType.DATE_RANGE,
            'in_list': FilterType.CATEGORICAL,
            'equals_string': FilterType.CATEGORICAL,
            'like_pattern': FilterType.CATEGORICAL,
            'greater_than': FilterType.NUMERIC_RANGE,
            'less_than': FilterType.NUMERIC_RANGE,
            'equals_number': FilterType.NUMERIC_RANGE
        }
        
        filter_type = filter_type_map.get(condition_dict['filter_type'], FilterType.CATEGORICAL)
        
        # Check if parameter-driven
        is_parameter_driven = '@' in condition
        parameter_name = None
        if is_parameter_driven:
            param_match = re.search(r'@(\w+)', condition)
            parameter_name = param_match.group(1) if param_match else None
        
        return BusinessFilter(
            field_name=field_name,
            filter_type=filter_type,
            condition=condition,
            values=values,
            is_parameter_driven=is_parameter_driven,
            parameter_name=parameter_name
        )
    
    @staticmethod
    def parse_calculation(name: str, expression: str) -> Optional[BusinessCalculation]:
        """Parse RDL expression to extract calculation logic"""
        if not expression or not expression.startswith('='):
            return None
        
        expression = expression[1:]  # Remove leading =
        
        # Aggregation patterns
        agg_patterns = [
            (r'Sum\s*\(\s*Fields!(\w+)\.Value\s*\)', CalculationType.AGGREGATION, 'SUM'),
            (r'Count\s*\(\s*Fields!(\w+)\.Value\s*\)', CalculationType.AGGREGATION, 'COUNT'),
            (r'Avg\s*\(\s*Fields!(\w+)\.Value\s*\)', CalculationType.AGGREGATION, 'AVG'),
            (r'Max\s*\(\s*Fields!(\w+)\.Value\s*\)', CalculationType.AGGREGATION, 'MAX'),
            (r'Min\s*\(\s*Fields!(\w+)\.Value\s*\)', CalculationType.AGGREGATION, 'MIN')
        ]
        
        # Formula patterns
        formula_patterns = [
            (r'Fields!(\w+)\.Value\s*[\+\-\*\/]\s*Fields!(\w+)\.Value', CalculationType.FORMULA, 'ARITHMETIC'),
            (r'Fields!(\w+)\.Value\s*[\+\-\*\/]\s*[\d\.]+', CalculationType.FORMULA, 'ARITHMETIC'),
            (r'IIF\s*\(', CalculationType.CONDITIONAL, 'CONDITIONAL')
        ]
        
        # Date function patterns
        date_patterns = [
            (r'Format\s*\(\s*Fields!(\w+)\.Value\s*,\s*["\']([^"\']+)["\']\s*\)', CalculationType.DATE_FUNCTION, 'FORMAT'),
            (r'Year\s*\(\s*Fields!(\w+)\.Value\s*\)', CalculationType.DATE_FUNCTION, 'YEAR'),
            (r'Month\s*\(\s*Fields!(\w+)\.Value\s*\)', CalculationType.DATE_FUNCTION, 'MONTH')
        ]
        
        all_patterns = agg_patterns + formula_patterns + date_patterns
        
        for pattern, calc_type, agg_level in all_patterns:
            match = re.search(pattern, expression, re.IGNORECASE)
            if match:
                # Extract field names used in calculation
                field_matches = re.findall(r'Fields!(\w+)\.Value', expression)
                
                return BusinessCalculation(
                    name=name,
                    calculation_type=calc_type,
                    expression=expression,
                    fields_used=field_matches,
                    aggregation_level=agg_level
                )
        
        return None
    
    @staticmethod
    def determine_report_type(root: ET.Element) -> str:
        """Determine the type of report based on its structure"""
        namespace_map = {
            'rdl': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition'
        }
        
        # Check for different report item types
        has_tablix = len(root.findall('.//rdl:Tablix', namespace_map)) > 0
        has_chart = len(root.findall('.//rdl:Chart', namespace_map)) > 0
        has_matrix = False
        
        # Check if tablix is configured as matrix
        for tablix in root.findall('.//rdl:Tablix', namespace_map):
            col_groups = tablix.findall('.//rdl:TablixColumnHierarchy', namespace_map)
            row_groups = tablix.findall('.//rdl:TablixRowHierarchy', namespace_map)
            if col_groups and row_groups:
                has_matrix = True
                break
        
        if has_chart:
            return 'chart_report'
        elif has_matrix:
            return 'matrix_report'
        elif has_tablix:
            return 'tabular_report'
        else:
            return 'custom_report'
    
    @staticmethod
    def extract_output_fields(root: ET.Element) -> List[str]:
        """Extract the fields that are actually displayed in the report"""
        namespace_map = {
            'rdl': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition'
        }
        
        fields = []
        
        # Extract from textboxes that reference fields
        for textbox in root.findall('.//rdl:Textbox', namespace_map):
            value_elem = textbox.find('.//rdl:Value', namespace_map)
            if value_elem is not None and value_elem.text:
                # Look for field references
                field_matches = re.findall(r'Fields!(\w+)\.Value', value_elem.text)
                fields.extend(field_matches)
        
        # Remove duplicates and return
        return list(set(fields))
    
    @staticmethod
    def extract_grouping_fields(root: ET.Element) -> List[str]:
        """Extract fields used for grouping"""
        namespace_map = {
            'rdl': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition'
        }
        
        grouping_fields = []
        
        # Look for group expressions in tablix
        for group in root.findall('.//rdl:Group', namespace_map):
            for group_expr in group.findall('.//rdl:GroupExpression', namespace_map):
                if group_expr.text:
                    field_matches = re.findall(r'Fields!(\w+)\.Value', group_expr.text)
                    grouping_fields.extend(field_matches)
        
        return list(set(grouping_fields))
    
    @staticmethod
    def extract_sorting_fields(root: ET.Element) -> List[str]:
        """Extract fields used for sorting"""
        namespace_map = {
            'rdl': 'http://schemas.microsoft.com/sqlserver/reporting/2008/01/reportdefinition'
        }
        
        sorting_fields = []
        
        # Look for sort expressions
        for sort_expr in root.findall('.//rdl:SortExpression', namespace_map):
            if sort_expr.text:
                field_matches = re.findall(r'Fields!(\w+)\.Value', sort_expr.text)
                sorting_fields.extend(field_matches)
        
        return list(set(sorting_fields))
    
    @staticmethod
    def normalize_filters(filters: List[BusinessFilter]) -> List[Dict[str, Any]]:
        """Normalize filters for comparison"""
        normalized = []
        
        for filter_obj in filters:
            normalized_filter = {
                'field_name': filter_obj.field_name.lower(),
                'filter_type': filter_obj.filter_type.value,
                'condition_type': RDLParserHelpers._extract_condition_type(filter_obj.condition),
                'is_parameter_driven': filter_obj.is_parameter_driven,
                'value_count': len(filter_obj.values)
            }
            normalized.append(normalized_filter)
        
        return normalized
    
    @staticmethod
    def _extract_condition_type(condition: str) -> str:
        """Extract the type of condition (equals, range, in, like, etc.)"""
        condition_lower = condition.lower()
        
        if 'between' in condition_lower:
            return 'range'
        elif ' in ' in condition_lower:
            return 'in_list'
        elif 'like' in condition_lower:
            return 'pattern_match'
        elif '>=' in condition_lower:
            return 'greater_equal'
        elif '>' in condition_lower:
            return 'greater_than'
        elif '<=' in condition_lower:
            return 'less_equal'
        elif '<' in condition_lower:
            return 'less_than'
        elif '=' in condition_lower:
            return 'equals'
        else:
            return 'unknown'
    
    @staticmethod
    def calculate_filter_similarity(filters1: List[Dict[str, Any]], 
                                  filters2: List[Dict[str, Any]]) -> float:
        """Calculate similarity between two sets of normalized filters"""
        
        if not filters1 and not filters2:
            return 100.0
        
        if not filters1 or not filters2:
            return 0.0
        
        # Create filter signatures for comparison
        sig1 = {f"{f['field_name']}:{f['condition_type']}" for f in filters1}
        sig2 = {f"{f['field_name']}:{f['condition_type']}" for f in filters2}
        
        # Calculate Jaccard similarity
        intersection = len(sig1 & sig2)
        union = len(sig1 | sig2)
        
        return (intersection / union * 100) if union > 0 else 0.0
    
    @staticmethod
    def compare_lists(list1: List[str], list2: List[str]) -> float:
        """Compare two lists and return similarity percentage"""
        if not list1 and not list2:
            return 100.0
        
        if not list1 or not list2:
            return 0.0
        
        set1 = set(item.lower() for item in list1)
        set2 = set(item.lower() for item in list2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return (intersection / union * 100) if union > 0 else 0.0
    
    @staticmethod
    def infer_parameter_purpose(param_name: str) -> str:
        """Infer the business purpose of a parameter from its name"""
        name_lower = param_name.lower()
        
        if any(word in name_lower for word in ['date', 'from', 'to', 'start', 'end']):
            return 'date_filter'
        elif any(word in name_lower for word in ['region', 'location', 'office', 'branch']):
            return 'location_filter'
        elif any(word in name_lower for word in ['product', 'item', 'category']):
            return 'product_filter'
        elif any(word in name_lower for word in ['customer', 'client', 'account']):
            return 'customer_filter'
        elif any(word in name_lower for word in ['employee', 'staff', 'user']):
            return 'employee_filter'
        else:
            return 'general_filter'