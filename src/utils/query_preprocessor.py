"""
Query Preprocessor for Normalization
===================================
Advanced SQL query preprocessing and normalization for consistent analysis.
Handles multiple SQL dialects, removes noise, and standardizes queries
for accurate similarity comparison.

Features:
- Multi-dialect SQL parsing and normalization
- Parameter extraction and standardization
- Comment and whitespace cleanup
- Identifier normalization (case, quotes, brackets)
- Query structure canonicalization
- Business logic extraction
"""

import re
import logging
import sqlparse
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class QueryFeatures:
    """Extracted features from a normalized query"""
    original_query: str
    normalized_query: str
    canonical_query: str
    tables: List[str]
    columns: List[str]
    parameters: List[str]
    functions: List[str]
    operations: List[str]
    joins: List[str]
    filters: List[str]
    aggregations: List[str]
    sorting: List[str]
    grouping: List[str]
    complexity_score: float
    query_type: str
    business_intent: str
    semantic_signature: str

class QueryType(Enum):
    """Types of SQL queries"""
    SELECT = "SELECT"
    INSERT = "INSERT"  
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    DROP = "DROP"
    STORED_PROC = "STORED_PROCEDURE"
    UNKNOWN = "UNKNOWN"

class QueryPreprocessor:
    """
    Advanced SQL query preprocessor with multi-dialect support.
    Normalizes queries for consistent comparison and feature extraction.
    """
    
    def __init__(self):
        self.stop_words = self._initialize_stop_words()
        self.normalization_rules = self._initialize_normalization_rules()
        self.parameter_patterns = self._initialize_parameter_patterns()
        self.function_mappings = self._initialize_function_mappings()
        
    def preprocess_query(self, query: str, dialect: str = "auto") -> QueryFeatures:
        """
        Comprehensive query preprocessing and feature extraction.
        
        Args:
            query: Raw SQL query text
            dialect: SQL dialect hint ("auto", "tsql", "plsql", "postgresql", etc.)
            
        Returns:
            QueryFeatures object with normalized query and extracted features
        """
        logger.debug(f"Preprocessing query: {query[:100]}...")
        
        # Step 1: Basic cleanup
        cleaned_query = self._basic_cleanup(query)
        
        # Step 2: Normalize syntax
        normalized_query = self._normalize_syntax(cleaned_query, dialect)
        
        # Step 3: Extract parameters
        parameters, param_normalized = self._extract_and_normalize_parameters(normalized_query)
        
        # Step 4: Parse query structure
        parsed_query = self._parse_query(param_normalized)
        
        # Step 5: Extract structural features
        features = self._extract_structural_features(parsed_query)
        
        # Step 6: Create canonical representation
        canonical_query = self._canonicalize_query(param_normalized, features)
        
        # Step 7: Calculate complexity and intent
        complexity_score = self._calculate_complexity_score(features)
        query_type = self._determine_query_type(parsed_query)
        business_intent = self._infer_business_intent(features)
        
        # Step 8: Generate semantic signature
        semantic_signature = self._generate_semantic_signature(features)
        
        return QueryFeatures(
            original_query=query,
            normalized_query=normalized_query,
            canonical_query=canonical_query,
            tables=features['tables'],
            columns=features['columns'],
            parameters=parameters,
            functions=features['functions'],
            operations=features['operations'],
            joins=features['joins'],
            filters=features['filters'],
            aggregations=features['aggregations'],
            sorting=features['sorting'],
            grouping=features['grouping'],
            complexity_score=complexity_score,
            query_type=query_type.value,
            business_intent=business_intent,
            semantic_signature=semantic_signature
        )
    
    def batch_preprocess(self, queries: Dict[str, str]) -> Dict[str, QueryFeatures]:
        """
        Preprocess multiple queries efficiently.
        
        Args:
            queries: Dictionary mapping query_id to query text
            
        Returns:
            Dictionary mapping query_id to QueryFeatures
        """
        results = {}
        
        logger.info(f"Batch preprocessing {len(queries)} queries")
        
        for query_id, query_text in queries.items():
            try:
                features = self.preprocess_query(query_text)
                results[query_id] = features
            except Exception as e:
                logger.error(f"Error preprocessing query {query_id}: {e}")
                # Create minimal features for failed queries
                results[query_id] = self._create_fallback_features(query_text)
        
        return results
    
    def compare_query_features(self, features1: QueryFeatures, 
                             features2: QueryFeatures) -> Dict[str, float]:
        """
        Compare two sets of query features for similarity.
        
        Args:
            features1: First query features
            features2: Second query features
            
        Returns:
            Dictionary of similarity scores for different dimensions
        """
        similarities = {}
        
        # Table similarity
        similarities['table_similarity'] = self._jaccard_similarity(
            set(features1.tables), set(features2.tables)
        )
        
        # Column similarity
        similarities['column_similarity'] = self._jaccard_similarity(
            set(features1.columns), set(features2.columns)
        )
        
        # Function similarity
        similarities['function_similarity'] = self._jaccard_similarity(
            set(features1.functions), set(features2.functions)
        )
        
        # Operation similarity
        similarities['operation_similarity'] = self._jaccard_similarity(
            set(features1.operations), set(features2.operations)
        )
        
        # Join pattern similarity
        similarities['join_similarity'] = self._jaccard_similarity(
            set(features1.joins), set(features2.joins)
        )
        
        # Business intent similarity
        similarities['intent_similarity'] = (
            1.0 if features1.business_intent == features2.business_intent else 0.0
        )
        
        # Query type similarity
        similarities['type_similarity'] = (
            1.0 if features1.query_type == features2.query_type else 0.0
        )
        
        # Complexity similarity
        complexity_diff = abs(features1.complexity_score - features2.complexity_score)
        max_complexity = max(features1.complexity_score, features2.complexity_score)
        similarities['complexity_similarity'] = (
            1.0 - (complexity_diff / max(max_complexity, 1.0))
        )
        
        # Canonical query similarity (exact match)
        similarities['canonical_similarity'] = (
            1.0 if features1.canonical_query == features2.canonical_query else 0.0
        )
        
        # Semantic signature similarity
        similarities['semantic_similarity'] = (
            1.0 if features1.semantic_signature == features2.semantic_signature else 
            self._signature_similarity(features1.semantic_signature, features2.semantic_signature)
        )
        
        return similarities
    
    def _basic_cleanup(self, query: str) -> str:
        """Basic query cleanup"""
        # Remove line comments
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        
        # Remove block comments
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        query = query.strip()
        
        return query
    
    def _normalize_syntax(self, query: str, dialect: str) -> str:
        """Normalize SQL syntax for consistent parsing"""
        normalized = query
        
        # Apply dialect-specific normalizations
        for pattern, replacement in self.normalization_rules.get(dialect, {}).items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Apply general normalizations
        for pattern, replacement in self.normalization_rules.get('general', {}).items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Format with sqlparse for consistency
        try:
            normalized = sqlparse.format(
                normalized,
                keyword_case='upper',
                identifier_case='lower',
                strip_comments=True,
                reindent=False
            )
        except Exception as e:
            logger.warning(f"Error formatting query with sqlparse: {e}")
        
        return normalized
    
    def _extract_and_normalize_parameters(self, query: str) -> Tuple[List[str], str]:
        """Extract parameters and normalize them"""
        parameters = []
        normalized_query = query
        
        # Extract different parameter patterns
        for param_type, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    param_name = match[0] if match[0] else match[1]
                else:
                    param_name = match
                
                if param_name not in parameters:
                    parameters.append(param_name)
        
        # Normalize parameters to generic placeholders
        param_counter = 1
        for param in parameters:
            # Replace specific parameter with generic placeholder
            param_placeholder = f"@PARAM{param_counter}"
            
            # Handle different parameter formats
            patterns_to_replace = [
                f"@{param}",      # T-SQL style
                f":{param}",      # Oracle/PostgreSQL style
                f"${param}",      # PostgreSQL numbered
                f"?{param}?",     # Some other formats
            ]
            
            for pattern in patterns_to_replace:
                normalized_query = normalized_query.replace(pattern, param_placeholder)
            
            param_counter += 1
        
        return parameters, normalized_query
    
    def _parse_query(self, query: str) -> sqlparse.sql.Statement:
        """Parse query using sqlparse"""
        try:
            parsed = sqlparse.parse(query)
            return parsed[0] if parsed else None
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return None
    
    def _extract_structural_features(self, parsed_query) -> Dict[str, List[str]]:
        """Extract structural features from parsed query"""
        features = {
            'tables': [],
            'columns': [],
            'functions': [],
            'operations': [],
            'joins': [],
            'filters': [],
            'aggregations': [],
            'sorting': [],
            'grouping': []
        }
        
        if not parsed_query:
            return features
        
        # Extract features by traversing the parse tree
        self._traverse_tokens(parsed_query, features)
        
        # Remove duplicates and clean up
        for key in features:
            features[key] = list(set(features[key]))
            features[key] = [item for item in features[key] if item and item.strip()]
        
        return features
    
    def _traverse_tokens(self, token, features: Dict[str, List[str]]):
        """Recursively traverse SQL tokens to extract features"""
        if hasattr(token, 'tokens'):
            for subtoken in token.tokens:
                self._traverse_tokens(subtoken, features)
        
        # Extract different types of features
        token_value = str(token).strip()
        token_type = token.ttype
        
        if not token_value or token_value.isspace():
            return
        
        # Keywords and operations
        if token_type in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.Reserved):
            keyword = token_value.upper()
            
            if keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']:
                features['operations'].append(keyword)
            elif keyword in ['INNER', 'LEFT', 'RIGHT', 'OUTER', 'FULL', 'CROSS']:
                if 'JOIN' in str(token.parent).upper():
                    features['joins'].append(f"{keyword} JOIN")
            elif keyword == 'JOIN':
                features['joins'].append('JOIN')
            elif keyword in ['GROUP', 'ORDER']:
                next_token = self._get_next_significant_token(token)
                if next_token and str(next_token).upper() == 'BY':
                    if keyword == 'GROUP':
                        features['grouping'].append('GROUP BY')
                    else:
                        features['sorting'].append('ORDER BY')
            elif keyword in ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN', 'DISTINCT']:
                features['aggregations'].append(keyword)
        
        # Functions
        if token_type is None and '(' in token_value:
            func_match = re.match(r'(\w+)\s*\(', token_value)
            if func_match:
                func_name = func_match.group(1).upper()
                if func_name not in ['SELECT', 'FROM', 'WHERE']:
                    features['functions'].append(func_name)
        
        # Identifiers (tables, columns)
        if token_type in (sqlparse.tokens.Name, None) and token.ttype != sqlparse.tokens.Keyword:
            identifier = self._clean_identifier(token_value)
            if identifier:
                # Simple heuristic: if it's after FROM or JOIN, likely a table
                context = self._get_token_context(token)
                if any(ctx in context.upper() for ctx in ['FROM', 'JOIN']):
                    features['tables'].append(identifier)
                else:
                    # Check if it looks like a column reference
                    if '.' in identifier:
                        parts = identifier.split('.')
                        if len(parts) == 2:
                            features['tables'].append(parts[0])
                            features['columns'].append(parts[1])
                    else:
                        features['columns'].append(identifier)
        
        # Operators for filters
        if token_type in (sqlparse.tokens.Operator, sqlparse.tokens.Operator.Comparison):
            features['filters'].append(token_value)
    
    def _get_next_significant_token(self, token):
        """Get the next non-whitespace token"""
        if hasattr(token, 'parent') and hasattr(token.parent, 'tokens'):
            tokens = token.parent.tokens
            try:
                current_index = tokens.index(token)
                for i in range(current_index + 1, len(tokens)):
                    next_token = tokens[i]
                    if not str(next_token).isspace():
                        return next_token
            except (ValueError, IndexError):
                pass
        return None
    
    def _get_token_context(self, token) -> str:
        """Get context around a token for better classification"""
        context = ""
        if hasattr(token, 'parent'):
            context = str(token.parent)[:200]  # Get surrounding context
        return context
    
    def _clean_identifier(self, identifier: str) -> str:
        """Clean and normalize identifier names"""
        # Remove quotes, brackets, backticks
        cleaned = identifier.strip('[]"`\'')
        
        # Remove schema prefixes for simple comparison
        if '.' in cleaned:
            parts = cleaned.split('.')
            # Keep table.column format, but remove database.schema.table
            if len(parts) > 2:
                cleaned = '.'.join(parts[-2:])
        
        return cleaned.lower()
    
    def _canonicalize_query(self, query: str, features: Dict[str, List[str]]) -> str:
        """Create canonical representation of query"""
        # Sort all feature lists for consistent representation
        canonical_features = {}
        for key, value in features.items():
            canonical_features[key] = sorted(value)
        
        # Create a canonical structure representation
        canonical_parts = []
        
        if 'SELECT' in features.get('operations', []):
            canonical_parts.append('SELECT')
            canonical_parts.extend(sorted(features.get('columns', [])))
            canonical_parts.append('FROM')
            canonical_parts.extend(sorted(features.get('tables', [])))
            
            if features.get('joins'):
                canonical_parts.extend(sorted(features['joins']))
            
            if features.get('filters'):
                canonical_parts.append('WHERE')
                canonical_parts.extend(sorted(features['filters']))
            
            if features.get('grouping'):
                canonical_parts.extend(sorted(features['grouping']))
            
            if features.get('sorting'):
                canonical_parts.extend(sorted(features['sorting']))
        
        return ' '.join(canonical_parts)
    
    def _calculate_complexity_score(self, features: Dict[str, List[str]]) -> float:
        """Calculate query complexity score"""
        complexity = 0.0
        
        # Base complexity from structure
        complexity += len(features.get('tables', [])) * 1.0
        complexity += len(features.get('columns', [])) * 0.1
        complexity += len(features.get('joins', [])) * 2.0
        complexity += len(features.get('functions', [])) * 0.5
        complexity += len(features.get('aggregations', [])) * 1.5
        complexity += len(features.get('filters', [])) * 0.3
        
        # Additional complexity for advanced features
        if features.get('grouping'):
            complexity += 2.0
        if features.get('sorting'):
            complexity += 1.0
        
        return complexity
    
    def _determine_query_type(self, parsed_query) -> QueryType:
        """Determine the type of SQL query"""
        if not parsed_query:
            return QueryType.UNKNOWN
        
        query_str = str(parsed_query).upper().strip()
        
        if query_str.startswith('SELECT'):
            return QueryType.SELECT
        elif query_str.startswith('INSERT'):
            return QueryType.INSERT
        elif query_str.startswith('UPDATE'):
            return QueryType.UPDATE
        elif query_str.startswith('DELETE'):
            return QueryType.DELETE
        elif query_str.startswith('CREATE'):
            return QueryType.CREATE
        elif query_str.startswith('ALTER'):
            return QueryType.ALTER
        elif query_str.startswith('DROP'):
            return QueryType.DROP
        elif any(keyword in query_str for keyword in ['EXEC', 'EXECUTE', 'CALL']):
            return QueryType.STORED_PROC
        
        return QueryType.UNKNOWN
    
    def _infer_business_intent(self, features: Dict[str, List[str]]) -> str:
        """Infer business intent from query features"""
        
        # Analyze aggregations
        has_aggregations = bool(features.get('aggregations'))
        has_grouping = bool(features.get('grouping'))
        has_sorting = bool(features.get('sorting'))
        
        # Analyze table/column patterns
        tables = [t.lower() for t in features.get('tables', [])]
        columns = [c.lower() for c in features.get('columns', [])]
        
        # Business domain analysis
        if any(word in ' '.join(tables + columns) for word in ['customer', 'order', 'sale', 'product']):
            if has_aggregations and has_grouping:
                return 'sales_analytics'
            else:
                return 'sales_operations'
        
        if any(word in ' '.join(tables + columns) for word in ['employee', 'staff', 'department', 'salary']):
            return 'hr_analytics'
        
        if any(word in ' '.join(tables + columns) for word in ['inventory', 'stock', 'warehouse', 'item']):
            return 'inventory_management'
        
        # General patterns
        if has_aggregations and has_grouping:
            return 'analytical_reporting'
        elif has_aggregations:
            return 'summary_reporting'
        elif len(tables) > 3:
            return 'complex_operational'
        elif has_sorting:
            return 'listing_report'
        else:
            return 'general_query'
    
    def _generate_semantic_signature(self, features: Dict[str, List[str]]) -> str:
        """Generate semantic signature for quick similarity checks"""
        
        # Create a stable signature based on key features
        signature_elements = {
            'tables': sorted(features.get('tables', [])),
            'operations': sorted(features.get('operations', [])),
            'joins': sorted(features.get('joins', [])),
            'aggregations': sorted(features.get('aggregations', [])),
            'functions': sorted(features.get('functions', []))
        }
        
        signature_str = json.dumps(signature_elements, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between semantic signatures"""
        # For now, simple binary comparison
        # Could be enhanced with Hamming distance or other metrics
        return 1.0 if sig1 == sig2 else 0.0
    
    def _create_fallback_features(self, query: str) -> QueryFeatures:
        """Create minimal features for failed queries"""
        return QueryFeatures(
            original_query=query,
            normalized_query=query,
            canonical_query=query,
            tables=[],
            columns=[],
            parameters=[],
            functions=[],
            operations=[],
            joins=[],
            filters=[],
            aggregations=[],
            sorting=[],
            grouping=[],
            complexity_score=0.0,
            query_type=QueryType.UNKNOWN.value,
            business_intent='unknown',
            semantic_signature=hashlib.md5(query.encode()).hexdigest()
        )
    
    def _initialize_stop_words(self) -> Set[str]:
        """Initialize SQL stop words"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with'
        }
    
    def _initialize_normalization_rules(self) -> Dict[str, Dict[str, str]]:
        """Initialize normalization rules for different dialects"""
        return {
            'general': {
                r'\s+': ' ',  # Multiple spaces to single space
                r'\(\s+': '(',  # Remove space after opening parenthesis
                r'\s+\)': ')',  # Remove space before closing parenthesis
                r'\s*,\s*': ', ',  # Normalize comma spacing
                r'\s*=\s*': ' = ',  # Normalize equals spacing
            },
            'tsql': {
                r'\[([^\]]+)\]': r'"\1"',  # Convert brackets to quotes
                r'\bTOP\s+(\d+)': r'LIMIT \1',  # Convert TOP to LIMIT
                r'\bISNULL\s*\(': 'COALESCE(',  # Convert ISNULL to COALESCE
                r'\bGETDATE\s*\(\)': 'CURRENT_TIMESTAMP',  # Convert GETDATE
            },
            'mysql': {
                r'`([^`]+)`': r'"\1"',  # Convert backticks to quotes
                r'\bIFNULL\s*\(': 'COALESCE(',  # Convert IFNULL
                r'\bNOW\s*\(\)': 'CURRENT_TIMESTAMP',  # Convert NOW
            },
            'oracle': {
                r'\bNVL\s*\(': 'COALESCE(',  # Convert NVL
                r'\bSYSDATE\b': 'CURRENT_TIMESTAMP',  # Convert SYSDATE
                r'\bDUAL\b': 'DUMMY_TABLE',  # Normalize DUAL references
            }
        }
    
    def _initialize_parameter_patterns(self) -> Dict[str, str]:
        """Initialize parameter detection patterns"""
        return {
            'tsql': r'@(\w+)',  # @parameter
            'oracle': r':(\w+)',  # :parameter  
            'postgresql_named': r'\$(\w+)',  # $parameter
            'postgresql_numbered': r'\$(\d+)',  # $1, $2, etc.
            'generic': r'\?(\w*)',  # ? or ?name
        }
    
    def _initialize_function_mappings(self) -> Dict[str, str]:
        """Initialize function mappings for normalization"""
        return {
            # T-SQL to standard
            'ISNULL': 'COALESCE',
            'GETDATE': 'CURRENT_TIMESTAMP',
            'DATEDIFF': 'DATE_DIFF',
            'DATEADD': 'DATE_ADD',
            'LEN': 'LENGTH',
            
            # MySQL to standard
            'IFNULL': 'COALESCE',
            'NOW': 'CURRENT_TIMESTAMP',
            
            # Oracle to standard
            'NVL': 'COALESCE',
            'NVL2': 'CASE_COALESCE',
            'SYSDATE': 'CURRENT_TIMESTAMP',
        }

# Usage example and testing
if __name__ == "__main__":
    preprocessor = QueryPreprocessor()
    
    # Test queries
    test_queries = {
        'tsql_query': """
            DECLARE @StartDate DATETIME;
            SET @StartDate = GETDATE();
            SELECT TOP 10 c.CustomerID, ISNULL(o.OrderDate, GETDATE()) as OrderDate,
                   SUM(o.TotalAmount) as Total
            FROM [Customers] c
            INNER JOIN [Orders] o ON c.CustomerID = o.CustomerID
            WHERE o.OrderDate >= @StartDate
            GROUP BY c.CustomerID, o.OrderDate
            ORDER BY Total DESC;
        """,
        
        'mysql_query': """
            SELECT `customer_id`, IFNULL(`order_date`, NOW()) as order_date,
                   COUNT(*) as order_count
            FROM `orders` o
            LEFT JOIN `customers` c ON o.customer_id = c.id  
            WHERE o.status = 'completed'
            GROUP BY `customer_id`, `order_date`
            LIMIT 50;
        """,
        
        'postgresql_query': """
            SELECT generate_series($1, $2) AS month_id,
                   COALESCE(o.order_date, CURRENT_TIMESTAMP) as order_date,
                   COUNT(o.order_id) as orders
            FROM orders o
            WHERE o.created_at >= $3
            GROUP BY month_id, order_date
            ORDER BY month_id;
        """
    }
    
    print("Query Preprocessing Results:")
    print("=" * 60)
    
    # Process each query
    results = preprocessor.batch_preprocess(test_queries)
    
    for query_id, features in results.items():
        print(f"\nQuery: {query_id}")
        print(f"Type: {features.query_type}")
        print(f"Intent: {features.business_intent}")
        print(f"Complexity: {features.complexity_score:.2f}")
        print(f"Tables: {', '.join(features.tables[:5])}...")
        print(f"Functions: {', '.join(features.functions[:5])}...")
        print(f"Parameters: {', '.join(features.parameters)}")
        print(f"Signature: {features.semantic_signature[:16]}...")
    
    # Compare two queries
    if len(results) >= 2:
        keys = list(results.keys())
        similarities = preprocessor.compare_query_features(results[keys[0]], results[keys[1]])
        print(f"\nSimilarity between {keys[0]} and {keys[1]}:")
        for metric, score in similarities.items():
            print(f"  {metric}: {score:.3f}")