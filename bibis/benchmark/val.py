from dataclasses import dataclass

@dataclass
class ValidationResult:
    warnings: list[str]
    errors: list[str]