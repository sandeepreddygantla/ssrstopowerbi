"""
RDL Types and Data Classes
=========================
Shared data structures for RDL business logic analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

class FilterType(Enum):
    DATE_RANGE = "date_range"
    CATEGORICAL = "categorical" 
    NUMERIC_RANGE = "numeric_range"
    BOOLEAN = "boolean"
    PARAMETER_BASED = "parameter_based"

class CalculationType(Enum):
    AGGREGATION = "aggregation"  # SUM, COUNT, AVG
    FORMULA = "formula"          # Custom calculations
    CONDITIONAL = "conditional"   # CASE WHEN logic
    DATE_FUNCTION = "date_function"  # Date manipulations

@dataclass
class BusinessFilter:
    """Represents a business rule/filter in the report"""
    field_name: str
    filter_type: FilterType
    condition: str
    values: List[str]
    is_parameter_driven: bool = False
    parameter_name: Optional[str] = None

@dataclass
class BusinessCalculation:
    """Represents a calculation or aggregation in the report"""
    name: str
    calculation_type: CalculationType
    expression: str
    fields_used: List[str]
    aggregation_level: Optional[str] = None

@dataclass
class BusinessOutput:
    """Represents what the report actually outputs"""
    fields: List[str]
    grouping_fields: List[str]
    sorting_fields: List[str]
    display_format: str
    report_type: str  # tabular, matrix, chart, etc.

@dataclass
class BusinessContext:
    """Complete business context of an RDL report"""
    data_sources: List[str]
    filters: List[BusinessFilter]
    calculations: List[BusinessCalculation]
    output: BusinessOutput
    parameters: Dict[str, Any]
    business_purpose: str  # Inferred from structure