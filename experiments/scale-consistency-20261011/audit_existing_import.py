"""Import the fixed offline pair statistic from its hyphenated script."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('existing_audit', Path(__file__).with_name('audit-existing.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
measure = module.measure
