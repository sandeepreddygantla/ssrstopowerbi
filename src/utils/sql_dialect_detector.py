"""
SQL Dialect Detection and Normalization
=======================================
Advanced SQL dialect detector supporting all SSRS-compatible SQL dialects
with normalization capabilities for consistent comparison analysis.

Supported Dialects:
- T-SQL (SQL Server)
- PL/SQL (Oracle) 
- PostgreSQL
- MySQL
- SQLite
- IBM DB2
- Standard ANSI SQL
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SQLDialect(Enum):
    """Supported SQL dialects"""
    TSQL = "T-SQL"
    PLSQL = "PL/SQL" 
    POSTGRESQL = "PostgreSQL"
    MYSQL = "MySQL"
    SQLITE = "SQLite"
    DB2 = "DB2"
    ORACLE = "Oracle"
    ANSI_SQL = "ANSI SQL"
    UNKNOWN = "Unknown"

@dataclass
class DialectDetectionResult:
    """Result of dialect detection"""
    primary_dialect: SQLDialect
    confidence: float
    dialect_scores: Dict[SQLDialect, float]
    detected_features: List[str]
    normalization_notes: List[str]

class SQLDialectDetector:
    """
    Advanced SQL dialect detector with normalization capabilities.
    Uses pattern matching and feature analysis to identify SQL dialects.
    """
    
    def __init__(self):
        self.dialect_patterns = self._initialize_dialect_patterns()
        self.function_mappings = self._initialize_function_mappings()
        self.syntax_normalizations = self._initialize_syntax_normalizations()
    
    def detect_dialect(self, query: str) -> DialectDetectionResult:
        """
        Detect SQL dialect from query text with confidence scoring.
        
        Args:
            query: SQL query text to analyze
            
        Returns:
            DialectDetectionResult with primary dialect and confidence
        """
        query_upper = query.upper()
        dialect_scores = {}
        detected_features = []
        
        # Score each dialect based on pattern matches
        for dialect, patterns in self.dialect_patterns.items():
            score = 0
            matched_features = []
            
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                weight = pattern_info['weight']
                feature_name = pattern_info['name']
                
                if re.search(pattern, query_upper):
                    score += weight
                    matched_features.append(feature_name)
                    detected_features.extend(matched_features)
            
            dialect_scores[dialect] = score
        
        # Find primary dialect (highest score)
        if dialect_scores:
            primary_dialect = max(dialect_scores, key=dialect_scores.get)
            max_score = dialect_scores[primary_dialect]
            
            # Calculate confidence based on score distribution
            total_score = sum(dialect_scores.values())
            confidence = (max_score / total_score) if total_score > 0 else 0.0
            
            # Adjust confidence based on certainty indicators
            confidence = self._adjust_confidence(query_upper, primary_dialect, confidence)
        else:
            primary_dialect = SQLDialect.ANSI_SQL
            confidence = 0.5
        
        return DialectDetectionResult(
            primary_dialect=primary_dialect,
            confidence=confidence,
            dialect_scores=dialect_scores,
            detected_features=list(set(detected_features)),
            normalization_notes=[]
        )
    
    def normalize_query(self, query: str, target_dialect: SQLDialect = SQLDialect.ANSI_SQL) -> Tuple[str, List[str]]:
        """
        Normalize query to target dialect for consistent comparison.
        
        Args:
            query: Original SQL query
            target_dialect: Target dialect for normalization
            
        Returns:
            Tuple of (normalized_query, normalization_notes)
        """
        normalized = query
        normalization_notes = []
        
        # Detect source dialect
        detection_result = self.detect_dialect(query)
        source_dialect = detection_result.primary_dialect
        
        logger.info(f"Normalizing {source_dialect.value} query to {target_dialect.value}")
        
        # Apply normalization rules
        if source_dialect != target_dialect:
            normalized, notes = self._apply_normalization_rules(
                normalized, source_dialect, target_dialect
            )
            normalization_notes.extend(notes)
        
        # Apply general cleanup
        normalized = self._general_normalization(normalized)
        
        return normalized, normalization_notes
    
    def _initialize_dialect_patterns(self) -> Dict[SQLDialect, List[Dict]]:
        """Initialize dialect detection patterns with weights"""
        return {
            SQLDialect.TSQL: [
                {'pattern': r'\bDECLARE\s+@\w+', 'weight': 10, 'name': 'T-SQL Variables'},
                {'pattern': r'\bSET\s+@\w+', 'weight': 8, 'name': 'T-SQL Variable Assignment'},
                {'pattern': r'\b@@\w+', 'weight': 10, 'name': 'T-SQL Global Variables'},
                {'pattern': r'\bEXEC\b|\bEXECUTE\b', 'weight': 6, 'name': 'T-SQL Execute'},
                {'pattern': r'\bSP_\w+', 'weight': 8, 'name': 'T-SQL System Procedures'},
                {'pattern': r'\bISNULL\s*\(', 'weight': 7, 'name': 'T-SQL ISNULL Function'},
                {'pattern': r'\bDATEADD\s*\(|\bDATEDIFF\s*\(', 'weight': 8, 'name': 'T-SQL Date Functions'},
                {'pattern': r'\bGETDATE\s*\(\)', 'weight': 9, 'name': 'T-SQL GETDATE'},
                {'pattern': r'\bTOP\s+\d+', 'weight': 7, 'name': 'T-SQL TOP Clause'},
                {'pattern': r'\bOUTPUT\b', 'weight': 8, 'name': 'T-SQL OUTPUT Clause'},
                {'pattern': r'\bMERGE\b', 'weight': 6, 'name': 'T-SQL MERGE Statement'},
                {'pattern': r'\[[\w\s]+\]', 'weight': 5, 'name': 'T-SQL Bracket Identifiers'},
                {'pattern': r'\bTRY\b.*\bCATCH\b', 'weight': 9, 'name': 'T-SQL Try-Catch'},
                {'pattern': r'\bRAISERROR\b', 'weight': 8, 'name': 'T-SQL RAISERROR'},
            ],
            
            SQLDialect.PLSQL: [
                {'pattern': r'\bDECLARE\s*$', 'weight': 8, 'name': 'PL/SQL Declare Block'},
                {'pattern': r'\bBEGIN\b.*\bEND\s*;', 'weight': 10, 'name': 'PL/SQL Begin-End Block'},
                {'pattern': r'\bEXCEPTION\b', 'weight': 10, 'name': 'PL/SQL Exception Handling'},
                {'pattern': r'\bCURSOR\b', 'weight': 9, 'name': 'PL/SQL Cursor'},
                {'pattern': r'\bLOOP\b|\bEND\s+LOOP\b', 'weight': 8, 'name': 'PL/SQL Loop'},
                {'pattern': r'\bIF\b.*\bTHEN\b.*\bEND\s+IF\b', 'weight': 8, 'name': 'PL/SQL If-Then'},
                {'pattern': r'\bFOR\b.*\bIN\b.*\bLOOP\b', 'weight': 8, 'name': 'PL/SQL For Loop'},
                {'pattern': r'\bPROCEDURE\b|\bFUNCTION\b', 'weight': 9, 'name': 'PL/SQL Procedures/Functions'},
                {'pattern': r'\bPKG_\w+', 'weight': 7, 'name': 'PL/SQL Package References'},
                {'pattern': r'\bNVL\s*\(|\bNVL2\s*\(', 'weight': 8, 'name': 'PL/SQL NVL Functions'},
                {'pattern': r'\bSYSDATE\b', 'weight': 8, 'name': 'PL/SQL SYSDATE'},
                {'pattern': r'\bDUAL\b', 'weight': 9, 'name': 'Oracle DUAL Table'},
                {'pattern': r'\bROWNUM\b', 'weight': 8, 'name': 'Oracle ROWNUM'},
                {'pattern': r'\bCONNECT\s+BY\b', 'weight': 9, 'name': 'Oracle Hierarchical Query'},
            ],
            
            SQLDialect.POSTGRESQL: [
                {'pattern': r'\$\d+', 'weight': 10, 'name': 'PostgreSQL Parameter Placeholders'},
                {'pattern': r'\$\w+\$', 'weight': 9, 'name': 'PostgreSQL Dollar Quoting'},
                {'pattern': r'\bRETURNING\b', 'weight': 8, 'name': 'PostgreSQL RETURNING Clause'},
                {'pattern': r'::', 'weight': 7, 'name': 'PostgreSQL Type Cast'},
                {'pattern': r'\bGENERATE_SERIES\s*\(', 'weight': 9, 'name': 'PostgreSQL Generate Series'},
                {'pattern': r'\bARRAY\[', 'weight': 8, 'name': 'PostgreSQL Arrays'},
                {'pattern': r'\bUNNEST\s*\(', 'weight': 8, 'name': 'PostgreSQL UNNEST'},
                {'pattern': r'\bCOALESCE\s*\(', 'weight': 6, 'name': 'PostgreSQL COALESCE'},
                {'pattern': r'\bEXTRACT\s*\(', 'weight': 6, 'name': 'PostgreSQL EXTRACT'},
                {'pattern': r'\bON\s+CONFLICT\b', 'weight': 9, 'name': 'PostgreSQL ON CONFLICT'},
                {'pattern': r'\bWITH\s+RECURSIVE\b', 'weight': 8, 'name': 'PostgreSQL Recursive CTE'},
                {'pattern': r'\bILIKE\b', 'weight': 8, 'name': 'PostgreSQL ILIKE'},
                {'pattern': r'\bTABLESAMPLE\b', 'weight': 7, 'name': 'PostgreSQL TABLESAMPLE'},
            ],
            
            SQLDialect.MYSQL: [
                {'pattern': r'`[\w\s]+`', 'weight': 9, 'name': 'MySQL Backtick Identifiers'},
                {'pattern': r'\bLIMIT\s+\d+', 'weight': 8, 'name': 'MySQL LIMIT Clause'},
                {'pattern': r'\bOFFSET\s+\d+', 'weight': 6, 'name': 'MySQL OFFSET Clause'},
                {'pattern': r'\bIFNULL\s*\(', 'weight': 8, 'name': 'MySQL IFNULL Function'},
                {'pattern': r'\bCONCAT\s*\(', 'weight': 6, 'name': 'MySQL CONCAT'},
                {'pattern': r'\bDATE_FORMAT\s*\(', 'weight': 8, 'name': 'MySQL DATE_FORMAT'},
                {'pattern': r'\bNOW\s*\(\)', 'weight': 7, 'name': 'MySQL NOW Function'},
                {'pattern': r'\bREPLACE\s+INTO\b', 'weight': 9, 'name': 'MySQL REPLACE INTO'},
                {'pattern': r'\bINSERT\s+IGNORE\b', 'weight': 8, 'name': 'MySQL INSERT IGNORE'},
                {'pattern': r'\bON\s+DUPLICATE\s+KEY\b', 'weight': 9, 'name': 'MySQL ON DUPLICATE KEY'},
                {'pattern': r'\bSTRAIGHT_JOIN\b', 'weight': 8, 'name': 'MySQL STRAIGHT_JOIN'},
                {'pattern': r'\bFORCE\s+INDEX\b', 'weight': 7, 'name': 'MySQL Force Index'},
            ],
            
            SQLDialect.SQLITE: [
                {'pattern': r'\bAUTOINCREMENT\b', 'weight': 9, 'name': 'SQLite AUTOINCREMENT'},
                {'pattern': r'\bPRAGMA\b', 'weight': 10, 'name': 'SQLite PRAGMA'},
                {'pattern': r'\bATTACH\s+DATABASE\b', 'weight': 9, 'name': 'SQLite ATTACH DATABASE'},
                {'pattern': r'\bDETACH\s+DATABASE\b', 'weight': 9, 'name': 'SQLite DETACH DATABASE'},
                {'pattern': r'\bVACUUM\b', 'weight': 8, 'name': 'SQLite VACUUM'},
                {'pattern': r'\bREINDEX\b', 'weight': 7, 'name': 'SQLite REINDEX'},
                {'pattern': r'\bANALYZE\b', 'weight': 6, 'name': 'SQLite ANALYZE'},
                {'pattern': r'\bIF\s+NOT\s+EXISTS\b', 'weight': 5, 'name': 'SQLite IF NOT EXISTS'},
            ],
            
            SQLDialect.DB2: [
                {'pattern': r'\bFETCH\s+FIRST\s+\d+\s+ROWS?\s+ONLY\b', 'weight': 9, 'name': 'DB2 FETCH FIRST'},
                {'pattern': r'\bWITH\s+UR\b', 'weight': 8, 'name': 'DB2 WITH UR (Uncommitted Read)'},
                {'pattern': r'\bOPTIMIZE\s+FOR\s+\d+\s+ROWS?\b', 'weight': 8, 'name': 'DB2 OPTIMIZE FOR'},
                {'pattern': r'\bVALUES\s+NEXTVAL\s+FOR\b', 'weight': 9, 'name': 'DB2 Sequence NEXTVAL'},
                {'pattern': r'\bGENERATED\s+ALWAYS\b', 'weight': 7, 'name': 'DB2 Generated Column'},
                {'pattern': r'\bCONCAT\s+OPERATOR\s+\|\|', 'weight': 6, 'name': 'DB2 Concat Operator'},
            ]
        }
    
    def _initialize_function_mappings(self) -> Dict[SQLDialect, Dict[str, str]]:
        """Initialize function mappings for normalization"""
        return {
            SQLDialect.TSQL: {
                'ISNULL': 'COALESCE',
                'GETDATE': 'CURRENT_TIMESTAMP',
                'DATEDIFF': 'DATE_DIFF',
                'DATEADD': 'DATE_ADD',
                'LEN': 'LENGTH',
            },
            SQLDialect.PLSQL: {
                'NVL': 'COALESCE',
                'NVL2': 'CASE_COALESCE',
                'SYSDATE': 'CURRENT_TIMESTAMP',
                'LENGTH': 'LENGTH',
            },
            SQLDialect.MYSQL: {
                'IFNULL': 'COALESCE',
                'NOW': 'CURRENT_TIMESTAMP',
                'CONCAT': 'CONCAT',
            },
            SQLDialect.POSTGRESQL: {
                'COALESCE': 'COALESCE',
                'EXTRACT': 'EXTRACT',
            }
        }
    
    def _initialize_syntax_normalizations(self) -> Dict[str, str]:
        """Initialize syntax normalization rules"""
        return {
            # Identifier normalization
            r'\[([^\]]+)\]': r'"\1"',  # T-SQL brackets to quotes
            r'`([^`]+)`': r'"\1"',     # MySQL backticks to quotes
            
            # Limit clause normalization
            r'\bTOP\s+(\d+)\b': r'LIMIT \1',  # T-SQL TOP to LIMIT
            
            # String concatenation
            r'\|\|': ' + ',  # Change concatenation operators
            
            # Comment normalization
            r'--([^\n]*)': r'/* \1 */',  # Line comments to block comments
        }
    
    def _adjust_confidence(self, query: str, dialect: SQLDialect, base_confidence: float) -> float:
        """Adjust confidence based on certainty indicators"""
        
        # High certainty indicators
        high_certainty_patterns = {
            SQLDialect.TSQL: [r'\b@@\w+', r'\bDECLARE\s+@\w+'],
            SQLDialect.PLSQL: [r'\bBEGIN\b.*\bEXCEPTION\b', r'\bCURSOR\b'],
            SQLDialect.POSTGRESQL: [r'\$\d+', r'\$\w+\$'],
            SQLDialect.MYSQL: [r'`\w+`', r'\bON\s+DUPLICATE\s+KEY\b'],
        }
        
        if dialect in high_certainty_patterns:
            for pattern in high_certainty_patterns[dialect]:
                if re.search(pattern, query):
                    base_confidence = min(1.0, base_confidence * 1.2)
                    break
        
        return base_confidence
    
    def _apply_normalization_rules(self, query: str, source_dialect: SQLDialect, 
                                  target_dialect: SQLDialect) -> Tuple[str, List[str]]:
        """Apply dialect-specific normalization rules"""
        
        normalized = query
        notes = []
        
        # Apply function mappings
        if source_dialect in self.function_mappings:
            mappings = self.function_mappings[source_dialect]
            for source_func, target_func in mappings.items():
                pattern = r'\b' + source_func + r'\s*\('
                if re.search(pattern, normalized, re.IGNORECASE):
                    normalized = re.sub(pattern, target_func + '(', normalized, flags=re.IGNORECASE)
                    notes.append(f"Mapped {source_func} to {target_func}")
        
        # Apply syntax normalizations
        for pattern, replacement in self.syntax_normalizations.items():
            if re.search(pattern, normalized):
                normalized = re.sub(pattern, replacement, normalized)
                notes.append(f"Normalized syntax: {pattern} -> {replacement}")
        
        return normalized, notes
    
    def _general_normalization(self, query: str) -> str:
        """Apply general normalization for consistent formatting"""
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query).strip()
        
        # Normalize keywords to uppercase
        keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'GROUP BY', 'ORDER BY', 'HAVING', 'UNION', 'INTERSECT', 'EXCEPT',
            'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP',
            'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS'
        ]
        
        for keyword in keywords:
            pattern = r'\b' + keyword.lower() + r'\b'
            normalized = re.sub(pattern, keyword, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def get_dialect_info(self, dialect: SQLDialect) -> Dict[str, any]:
        """Get detailed information about a specific dialect"""
        
        dialect_info = {
            SQLDialect.TSQL: {
                'name': 'Transact-SQL (SQL Server)',
                'vendor': 'Microsoft',
                'key_features': ['Variables (@var)', 'System functions (@@)', 'TOP clause', 'ISNULL function'],
                'common_functions': ['GETDATE', 'ISNULL', 'DATEADD', 'DATEDIFF', 'LEN'],
            },
            SQLDialect.PLSQL: {
                'name': 'PL/SQL (Oracle)',
                'vendor': 'Oracle',
                'key_features': ['Procedural blocks', 'Exception handling', 'Cursors', 'DUAL table'],
                'common_functions': ['SYSDATE', 'NVL', 'NVL2', 'DECODE', 'ROWNUM'],
            },
            SQLDialect.POSTGRESQL: {
                'name': 'PostgreSQL',
                'vendor': 'PostgreSQL Global Development Group',
                'key_features': ['Parameter placeholders ($1)', 'RETURNING clause', 'Arrays', 'ILIKE operator'],
                'common_functions': ['GENERATE_SERIES', 'UNNEST', 'EXTRACT', 'COALESCE'],
            },
            SQLDialect.MYSQL: {
                'name': 'MySQL',
                'vendor': 'Oracle (MySQL)',
                'key_features': ['Backtick identifiers', 'LIMIT clause', 'INSERT IGNORE', 'ON DUPLICATE KEY'],
                'common_functions': ['NOW', 'IFNULL', 'CONCAT', 'DATE_FORMAT'],
            },
            SQLDialect.SQLITE: {
                'name': 'SQLite',
                'vendor': 'SQLite Development Team',
                'key_features': ['AUTOINCREMENT', 'PRAGMA statements', 'ATTACH DATABASE'],
                'common_functions': ['LENGTH', 'SUBSTR', 'REPLACE', 'DATETIME'],
            }
        }
        
        return dialect_info.get(dialect, {
            'name': dialect.value,
            'vendor': 'Unknown',
            'key_features': [],
            'common_functions': []
        })
    
    def batch_detect_dialects(self, queries: List[str]) -> List[DialectDetectionResult]:
        """Detect dialects for multiple queries efficiently"""
        
        results = []
        for i, query in enumerate(queries):
            try:
                result = self.detect_dialect(query)
                results.append(result)
                logger.debug(f"Query {i+1}: Detected {result.primary_dialect.value} (confidence: {result.confidence:.2f})")
            except Exception as e:
                logger.error(f"Error detecting dialect for query {i+1}: {str(e)}")
                results.append(DialectDetectionResult(
                    primary_dialect=SQLDialect.UNKNOWN,
                    confidence=0.0,
                    dialect_scores={},
                    detected_features=[],
                    normalization_notes=[]
                ))
        
        return results

# Usage examples
if __name__ == "__main__":
    detector = SQLDialectDetector()
    
    # Test queries in different dialects
    test_queries = [
        # T-SQL
        "DECLARE @StartDate DATETIME; SET @StartDate = GETDATE(); SELECT TOP 10 CustomerID, ISNULL(OrderDate, GETDATE()) FROM Orders WHERE OrderDate >= @StartDate",
        
        # PL/SQL  
        "DECLARE v_count NUMBER; BEGIN SELECT COUNT(*) INTO v_count FROM dual; EXCEPTION WHEN NO_DATA_FOUND THEN v_count := 0; END;",
        
        # PostgreSQL
        "SELECT generate_series($1, $2) AS id, EXTRACT(year FROM created_at) FROM orders WHERE status = ANY($3) RETURNING id",
        
        # MySQL
        "SELECT `customer_id`, IFNULL(`order_date`, NOW()) FROM `orders` LIMIT 10 OFFSET 20 ON DUPLICATE KEY UPDATE status = 'updated'",
        
        # SQLite
        "CREATE TABLE test (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT); PRAGMA table_info(test); VACUUM;"
    ]
    
    print("SQL Dialect Detection Results:")
    print("=" * 50)
    
    for i, query in enumerate(test_queries):
        result = detector.detect_dialect(query)
        print(f"\nQuery {i+1}:")
        print(f"Detected: {result.primary_dialect.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Features: {', '.join(result.detected_features[:3])}...")
        
        # Test normalization
        normalized, notes = detector.normalize_query(query)
        if notes:
            print(f"Normalization applied: {len(notes)} changes")