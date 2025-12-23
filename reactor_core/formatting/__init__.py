"""
Dataset formatting module for Night Shift Training Engine.

Converts RawInteraction data to training-ready formats:
- ChatML: {"role": "system/user/assistant", "content": "..."}
- Alpaca: {"instruction": "...", "input": "...", "output": "..."}
- Preference: DPO pairs {prompt, chosen, rejected}
"""

from reactor_core.formatting.base_formatter import (
    BaseFormatter,
    FormattedExample,
    OutputFormat,
)
from reactor_core.formatting.chatml_formatter import ChatMLFormatter
from reactor_core.formatting.alpaca_formatter import AlpacaFormatter
from reactor_core.formatting.quality_filter import QualityFilter, FilterResult
from reactor_core.formatting.dataset_builder import DatasetBuilder

__all__ = [
    "BaseFormatter",
    "FormattedExample",
    "OutputFormat",
    "ChatMLFormatter",
    "AlpacaFormatter",
    "QualityFilter",
    "FilterResult",
    "DatasetBuilder",
]
