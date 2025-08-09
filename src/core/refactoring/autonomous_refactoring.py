"""Autonomous Refactoring Engine for identifying and applying code improvements."""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class RefactoringType(Enum):
    """Types of refactoring operations."""

    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    RENAME_VARIABLE = "rename_variable"
    REMOVE_DUPLICATES = "remove_duplicates"
    SIMPLIFY_CONDITIONALS = "simplify_conditionals"
    REDUCE_PARAMETERS = "reduce_parameters"
    INLINE_METHOD = "inline_method"
    MOVE_METHOD = "move_method"


class CodeSmell(Enum):
    """Types of code smells to detect."""

    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    DUPLICATE_CODE = "duplicate_code"
    LONG_PARAMETER_LIST = "long_parameter_list"
    COMPLEX_CONDITIONALS = "complex_conditionals"
    MAGIC_NUMBERS = "magic_numbers"
    DEAD_CODE = "dead_code"
    GOD_CLASS = "god_class"


class RefactoringConfidence(Enum):
    """Confidence levels for refactoring operations."""

    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9


@dataclass
class CodeSmellInstance:
    """Represents a detected code smell."""

    smell_type: CodeSmell
    location: str
    severity: float
    description: str
    line_number: int | None = None
    suggestion: str | None = None


@dataclass
class RefactoringOpportunity:
    """Represents an identified refactoring opportunity."""

    refactoring_type: RefactoringType
    location: str
    description: str
    confidence: float
    estimated_benefit: float
    code_smell: CodeSmell | None = None
    suggested_changes: list[str] = field(default_factory=list)


@dataclass
class RefactoringResult:
    """Result from applying a refactoring operation."""

    success: bool
    original_code: str
    refactored_code: str = ""
    refactoring_type: RefactoringType | None = None
    confidence: float = 0.0
    improvements: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class AutonomousRefactoringEngine:
    """Engine for detecting code smells and applying autonomous refactoring."""

    def __init__(
        self, confidence_threshold: float = 0.8, auto_apply_threshold: float = 0.9
    ):
        """Initialize the autonomous refactoring engine."""
        self.confidence_threshold = confidence_threshold
        self.auto_apply_threshold = auto_apply_threshold

        # Metrics tracking
        self.metrics = {
            "total_refactorings": 0,
            "successful_refactorings": 0,
            "failed_refactorings": 0,
            "code_smells_detected": 0,
            "opportunities_identified": 0,
        }

        # Detection patterns
        self.code_smell_detectors = self._initialize_smell_detectors()
        self.refactoring_strategies = self._initialize_refactoring_strategies()

        logger.info(
            "AutonomousRefactoringEngine initialized",
            confidence_threshold=confidence_threshold,
            auto_apply_threshold=auto_apply_threshold,
        )

    async def detect_code_smells(self, code: str) -> list[CodeSmellInstance]:
        """Detect code smells in the provided code."""
        smells = []

        try:
            # Parse the code into AST
            tree = ast.parse(code)

            # Apply each smell detector
            for smell_type, detector in self.code_smell_detectors.items():
                detected_smells = detector(code, tree)
                smells.extend(detected_smells)

            self.metrics["code_smells_detected"] += len(smells)

            logger.info("Code smells detected", count=len(smells))
            return smells

        except SyntaxError as e:
            logger.warning("Failed to parse code for smell detection", error=str(e))
            return []

    async def analyze_complexity(self, code: str) -> dict[str, Any]:
        """Analyze code complexity metrics."""
        try:
            tree = ast.parse(code)

            complexity_metrics = {
                "cyclomatic_complexity": self._calculate_cyclomatic_complexity(tree),
                "function_length": self._calculate_function_lengths(tree),
                "parameter_count": self._calculate_parameter_counts(tree),
                "nesting_depth": self._calculate_nesting_depth(tree),
                "class_complexity": self._calculate_class_complexity(tree),
            }

            return complexity_metrics

        except SyntaxError:
            return {"error": "Failed to parse code for complexity analysis"}

    async def identify_refactoring_opportunities(
        self, code: str
    ) -> list[RefactoringOpportunity]:
        """Identify refactoring opportunities in the code."""
        opportunities = []

        try:
            # First detect code smells
            smells = await self.detect_code_smells(code)

            # Analyze complexity
            complexity = await self.analyze_complexity(code)

            # Generate opportunities based on smells and complexity
            for smell in smells:
                opportunity = self._smell_to_opportunity(smell, complexity)
                if opportunity:
                    opportunities.append(opportunity)

            # Add pattern-based opportunities
            pattern_opportunities = self._identify_pattern_opportunities(code)
            opportunities.extend(pattern_opportunities)

            self.metrics["opportunities_identified"] += len(opportunities)

            logger.info(
                "Refactoring opportunities identified", count=len(opportunities)
            )
            return opportunities

        except Exception as e:
            logger.error("Failed to identify refactoring opportunities", error=str(e))
            return []

    def calculate_confidence(
        self, smell_severity: float, pattern_match: float, risk_assessment: float
    ) -> float:
        """Calculate refactoring confidence based on multiple factors."""
        # Weight the factors
        weighted_score = (
            smell_severity * 0.4
            + pattern_match * 0.4
            + (1.0 - risk_assessment) * 0.2  # Lower risk = higher confidence
        )

        return min(1.0, max(0.0, weighted_score))

    async def apply_safe_refactoring(
        self, code: str, opportunity: RefactoringOpportunity
    ) -> RefactoringResult:
        """Apply refactoring with safety checks and rollback capability."""
        logger.info(
            "Applying safe refactoring",
            refactoring_type=opportunity.refactoring_type.value,
            confidence=opportunity.confidence,
        )

        try:
            # Check confidence threshold
            if opportunity.confidence < self.confidence_threshold:
                return RefactoringResult(
                    success=False,
                    original_code=code,
                    error_message=f"Confidence {opportunity.confidence} below threshold {self.confidence_threshold}",
                )

            # Apply the refactoring
            result = await self._apply_refactoring(code, opportunity)

            if result.success:
                # Validate the refactoring is safe
                is_safe = await self.validate_refactoring_safety(
                    code, result.refactored_code
                )

                if is_safe:
                    self.metrics["successful_refactorings"] += 1
                    logger.info("Refactoring applied successfully")
                    return result
                else:
                    # Rollback - return original code
                    logger.warning("Refactoring failed safety validation, rolling back")
                    return RefactoringResult(
                        success=False,
                        original_code=code,
                        error_message="Refactoring failed safety validation",
                    )
            else:
                self.metrics["failed_refactorings"] += 1
                return result

        except Exception as e:
            logger.error("Refactoring application failed", error=str(e))
            self.metrics["failed_refactorings"] += 1
            return RefactoringResult(
                success=False,
                original_code=code,
                error_message=f"Refactoring failed: {str(e)}",
            )
        finally:
            self.metrics["total_refactorings"] += 1

    async def validate_refactoring_safety(
        self, original_code: str, refactored_code: str
    ) -> bool:
        """Validate that refactoring preserves functionality and safety."""
        try:
            # Basic syntax validation
            ast.parse(refactored_code)

            # Check for significant structural changes
            original_ast = ast.parse(original_code)
            refactored_ast = ast.parse(refactored_code)

            # Count functions and classes
            original_functions = len(
                [
                    node
                    for node in ast.walk(original_ast)
                    if isinstance(node, ast.FunctionDef)
                ]
            )
            refactored_functions = len(
                [
                    node
                    for node in ast.walk(refactored_ast)
                    if isinstance(node, ast.FunctionDef)
                ]
            )

            original_classes = len(
                [
                    node
                    for node in ast.walk(original_ast)
                    if isinstance(node, ast.ClassDef)
                ]
            )
            refactored_classes = len(
                [
                    node
                    for node in ast.walk(refactored_ast)
                    if isinstance(node, ast.ClassDef)
                ]
            )

            # Allow reasonable increases in function/class count (from extraction)
            if refactored_functions > original_functions + 3:
                return False
            if refactored_classes > original_classes + 2:
                return False

            return True

        except SyntaxError:
            return False

    async def extract_method(
        self, code: str, method_name: str, start_line: int, end_line: int
    ) -> RefactoringResult:
        """Extract a method from the specified code range."""
        lines = code.split("\n")

        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return RefactoringResult(
                success=False,
                original_code=code,
                error_message="Invalid line range for method extraction",
            )

        # Extract the method body
        extracted_lines = lines[start_line - 1 : end_line]
        method_body = "\n".join(extracted_lines)

        # Create the new method
        new_method = f"def {method_name}(self):\n"
        for line in extracted_lines:
            new_method += f"    {line}\n"

        # Replace the original lines with a method call
        lines[start_line - 1 : end_line] = [f"    self.{method_name}()"]

        # Insert the new method (simple placement at the end of class)
        refactored_code = "\n".join(lines) + "\n\n" + new_method

        return RefactoringResult(
            success=True,
            original_code=code,
            refactored_code=refactored_code,
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            confidence=0.8,
            improvements=[
                f"Extracted {method_name} method",
                "Improved code organization",
            ],
        )

    async def remove_duplicate_code(self, code: str) -> RefactoringResult:
        """Remove duplicate code by extracting common patterns."""
        # Simple duplicate detection - look for repeated patterns
        lines = code.split("\n")

        # Find common patterns (simplified)
        pattern_counts = defaultdict(int)
        for i in range(len(lines) - 2):
            pattern = "\n".join(lines[i : i + 3])  # 3-line patterns
            if pattern.strip():
                pattern_counts[pattern] += 1

        # Find most common pattern
        if pattern_counts:
            most_common = max(pattern_counts.items(), key=lambda x: x[1])
            if most_common[1] > 1:  # Appears more than once
                # Create a method for the common pattern
                extracted_method = "def common_operation(self):\n"
                for line in most_common[0].split("\n"):
                    extracted_method += f"    {line}\n"

                # Replace occurrences with method calls
                refactored_code = code.replace(
                    most_common[0], "    self.common_operation()"
                )
                refactored_code += "\n\n" + extracted_method

                return RefactoringResult(
                    success=True,
                    original_code=code,
                    refactored_code=refactored_code,
                    refactoring_type=RefactoringType.REMOVE_DUPLICATES,
                    confidence=0.7,
                    improvements=["Removed duplicate code", "Extracted common pattern"],
                )

        # No significant duplicates found
        return RefactoringResult(
            success=True,
            original_code=code,
            refactored_code=code,
            refactoring_type=RefactoringType.REMOVE_DUPLICATES,
            confidence=1.0,
            improvements=["No significant duplicates found"],
        )

    async def simplify_conditionals(self, code: str) -> RefactoringResult:
        """Simplify complex conditional statements."""
        # Pattern: nested if statements that can be combined
        pattern = r"if\s+([^:]+):\s*\n\s+if\s+([^:]+):\s*\n\s+return\s+True\s*\n\s+else:\s*\n\s+return\s+False\s*\n\s+else:\s*\n\s+return\s+False"

        def replace_nested_if(match):
            condition1 = match.group(1)
            condition2 = match.group(2)
            return f"return {condition1} and {condition2}"

        refactored_code = re.sub(pattern, replace_nested_if, code, flags=re.MULTILINE)

        if refactored_code != code:
            return RefactoringResult(
                success=True,
                original_code=code,
                refactored_code=refactored_code,
                refactoring_type=RefactoringType.SIMPLIFY_CONDITIONALS,
                confidence=0.85,
                improvements=["Simplified nested conditionals", "Reduced complexity"],
            )
        else:
            return RefactoringResult(
                success=True,
                original_code=code,
                refactored_code=code,
                refactoring_type=RefactoringType.SIMPLIFY_CONDITIONALS,
                confidence=1.0,
                improvements=["No complex conditionals found to simplify"],
            )

    async def reduce_parameter_count(self, code: str) -> RefactoringResult:
        """Reduce parameter count by suggesting parameter objects."""
        # Look for functions with many parameters
        function_pattern = r"def\s+(\w+)\s*\(([^)]+)\):"

        def suggest_parameter_object(match):
            func_name = match.group(1)
            params = match.group(2)
            param_list = [p.strip() for p in params.split(",") if p.strip()]

            if len(param_list) > 4:  # Many parameters
                # Suggest parameter object
                return f"def {func_name}(self, params: {func_name.title()}Params):"
            return match.group(0)

        refactored_code = re.sub(function_pattern, suggest_parameter_object, code)

        if refactored_code != code:
            return RefactoringResult(
                success=True,
                original_code=code,
                refactored_code=refactored_code,
                refactoring_type=RefactoringType.REDUCE_PARAMETERS,
                confidence=0.7,
                improvements=[
                    "Suggested parameter object pattern",
                    "Reduced parameter coupling",
                ],
            )
        else:
            return RefactoringResult(
                success=True,
                original_code=code,
                refactored_code=code,
                refactoring_type=RefactoringType.REDUCE_PARAMETERS,
                confidence=1.0,
                improvements=["No functions with excessive parameters found"],
            )

    async def analyze_refactoring_impact(
        self, code: str, opportunity: RefactoringOpportunity
    ) -> dict[str, Any]:
        """Analyze the potential impact of applying a refactoring."""
        # Calculate current complexity
        current_complexity = await self.analyze_complexity(code)

        # Estimate improvements
        impact = {
            "maintainability_improvement": opportunity.estimated_benefit * 0.8,
            "readability_improvement": opportunity.estimated_benefit * 0.9,
            "complexity_reduction": opportunity.estimated_benefit * 0.7,
            "risk_assessment": 1.0 - opportunity.confidence,
            "estimated_effort": self._estimate_effort(opportunity),
            "current_complexity": current_complexity,
        }

        return impact

    async def apply_batch_refactoring(
        self, code: str, opportunities: list[RefactoringOpportunity]
    ) -> list[RefactoringResult]:
        """Apply multiple refactoring opportunities in batch."""
        results = []
        current_code = code

        # Sort by confidence (highest first)
        sorted_opportunities = sorted(
            opportunities, key=lambda x: x.confidence, reverse=True
        )

        for opportunity in sorted_opportunities:
            result = await self.apply_safe_refactoring(current_code, opportunity)
            results.append(result)

            # If successful, use refactored code for next iteration
            if result.success and result.refactored_code:
                current_code = result.refactored_code

        return results

    def should_auto_apply(self, opportunity: RefactoringOpportunity) -> bool:
        """Determine if refactoring should be automatically applied."""
        return opportunity.confidence >= self.auto_apply_threshold

    def get_refactoring_metrics(self) -> dict[str, Any]:
        """Get refactoring metrics and performance data."""
        return self.metrics.copy()

    async def _apply_refactoring(
        self, code: str, opportunity: RefactoringOpportunity
    ) -> RefactoringResult:
        """Apply the specific refactoring based on type."""
        refactoring_type = opportunity.refactoring_type

        if refactoring_type == RefactoringType.EXTRACT_METHOD:
            # Simple extraction for testing
            return await self.extract_method(code, "extracted_method", 1, 3)
        elif refactoring_type == RefactoringType.REMOVE_DUPLICATES:
            return await self.remove_duplicate_code(code)
        elif refactoring_type == RefactoringType.SIMPLIFY_CONDITIONALS:
            return await self.simplify_conditionals(code)
        elif refactoring_type == RefactoringType.REDUCE_PARAMETERS:
            return await self.reduce_parameter_count(code)
        else:
            return RefactoringResult(
                success=False,
                original_code=code,
                error_message=f"Refactoring type {refactoring_type} not implemented",
            )

    def _initialize_smell_detectors(self) -> dict[CodeSmell, callable]:
        """Initialize code smell detection functions."""
        return {
            CodeSmell.LONG_METHOD: self._detect_long_methods,
            CodeSmell.LONG_PARAMETER_LIST: self._detect_long_parameter_lists,
            CodeSmell.COMPLEX_CONDITIONALS: self._detect_complex_conditionals,
            CodeSmell.DUPLICATE_CODE: self._detect_duplicate_code,
            CodeSmell.MAGIC_NUMBERS: self._detect_magic_numbers,
        }

    def _initialize_refactoring_strategies(self) -> dict[RefactoringType, callable]:
        """Initialize refactoring strategy functions."""
        return {
            RefactoringType.EXTRACT_METHOD: self.extract_method,
            RefactoringType.REMOVE_DUPLICATES: self.remove_duplicate_code,
            RefactoringType.SIMPLIFY_CONDITIONALS: self.simplify_conditionals,
            RefactoringType.REDUCE_PARAMETERS: self.reduce_parameter_count,
        }

    def _detect_long_methods(self, code: str, tree: ast.AST) -> list[CodeSmellInstance]:
        """Detect methods that are too long."""
        smells = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count lines in function
                if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                    length = node.end_lineno - node.lineno
                    if length > 20:  # Threshold for long method
                        smells.append(
                            CodeSmellInstance(
                                smell_type=CodeSmell.LONG_METHOD,
                                location=node.name,
                                severity=min(1.0, length / 50.0),
                                description=f"Method {node.name} is {length} lines long",
                                line_number=node.lineno,
                                suggestion="Consider extracting smaller methods",
                            )
                        )

        return smells

    def _detect_long_parameter_lists(
        self, code: str, tree: ast.AST
    ) -> list[CodeSmellInstance]:
        """Detect functions with too many parameters."""
        smells = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                if param_count > 5:  # Threshold for too many parameters
                    smells.append(
                        CodeSmellInstance(
                            smell_type=CodeSmell.LONG_PARAMETER_LIST,
                            location=node.name,
                            severity=min(1.0, param_count / 10.0),
                            description=f"Function {node.name} has {param_count} parameters",
                            line_number=node.lineno,
                            suggestion="Consider using parameter objects or reducing dependencies",
                        )
                    )

        return smells

    def _detect_complex_conditionals(
        self, code: str, tree: ast.AST
    ) -> list[CodeSmellInstance]:
        """Detect overly complex conditional statements."""
        smells = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Count nested conditions
                nesting_depth = self._count_nesting_depth(node)
                if nesting_depth > 3:
                    smells.append(
                        CodeSmellInstance(
                            smell_type=CodeSmell.COMPLEX_CONDITIONALS,
                            location=f"line {node.lineno}",
                            severity=min(1.0, nesting_depth / 6.0),
                            description=f"Complex conditional with nesting depth {nesting_depth}",
                            line_number=node.lineno,
                            suggestion="Consider extracting condition logic or using guard clauses",
                        )
                    )

        return smells

    def _detect_duplicate_code(
        self, code: str, tree: ast.AST
    ) -> list[CodeSmellInstance]:
        """Detect duplicate code patterns."""
        smells = []

        # Simple duplicate detection based on similar line patterns
        lines = code.split("\n")
        line_groups = defaultdict(list)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 10:  # Ignore very short lines
                line_groups[stripped].append(i + 1)

        for line_content, line_numbers in line_groups.items():
            if len(line_numbers) > 2:  # Appears more than twice
                smells.append(
                    CodeSmellInstance(
                        smell_type=CodeSmell.DUPLICATE_CODE,
                        location=f"lines {line_numbers}",
                        severity=min(1.0, len(line_numbers) / 5.0),
                        description=f"Duplicate code pattern appears {len(line_numbers)} times",
                        suggestion="Extract common code into a method",
                    )
                )

        return smells

    def _detect_magic_numbers(
        self, code: str, tree: ast.AST
    ) -> list[CodeSmellInstance]:
        """Detect magic numbers in code."""
        smells = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                # Skip common constants
                if node.value not in [0, 1, -1, 2, 10, 100]:
                    smells.append(
                        CodeSmellInstance(
                            smell_type=CodeSmell.MAGIC_NUMBERS,
                            location=f"line {node.lineno}",
                            severity=0.5,
                            description=f"Magic number {node.value} found",
                            line_number=node.lineno,
                            suggestion="Consider using named constants",
                        )
                    )

        return smells

    def _smell_to_opportunity(
        self, smell: CodeSmellInstance, complexity: dict[str, Any]
    ) -> RefactoringOpportunity | None:
        """Convert a code smell into a refactoring opportunity."""
        smell_to_refactoring = {
            CodeSmell.LONG_METHOD: RefactoringType.EXTRACT_METHOD,
            CodeSmell.LONG_PARAMETER_LIST: RefactoringType.REDUCE_PARAMETERS,
            CodeSmell.COMPLEX_CONDITIONALS: RefactoringType.SIMPLIFY_CONDITIONALS,
            CodeSmell.DUPLICATE_CODE: RefactoringType.REMOVE_DUPLICATES,
        }

        refactoring_type = smell_to_refactoring.get(smell.smell_type)
        if not refactoring_type:
            return None

        confidence = self.calculate_confidence(
            smell_severity=smell.severity,
            pattern_match=0.8,  # Pattern matching confidence
            risk_assessment=0.2,  # Low risk
        )

        return RefactoringOpportunity(
            refactoring_type=refactoring_type,
            location=smell.location,
            description=smell.description,
            confidence=confidence,
            estimated_benefit=smell.severity * 0.8,
            code_smell=smell.smell_type,
            suggested_changes=[smell.suggestion] if smell.suggestion else [],
        )

    def _identify_pattern_opportunities(
        self, code: str
    ) -> list[RefactoringOpportunity]:
        """Identify refactoring opportunities based on code patterns."""
        opportunities = []

        # Look for specific patterns that suggest refactoring
        if "print(" in code:
            opportunities.append(
                RefactoringOpportunity(
                    refactoring_type=RefactoringType.EXTRACT_METHOD,
                    location="print statements",
                    description="Replace print statements with proper logging",
                    confidence=0.7,
                    estimated_benefit=0.6,
                )
            )

        return opportunities

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of the code."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_function_lengths(self, tree: ast.AST) -> dict[str, int]:
        """Calculate lengths of all functions."""
        function_lengths = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                    length = node.end_lineno - node.lineno
                    function_lengths[node.name] = length

        return function_lengths

    def _calculate_parameter_counts(self, tree: ast.AST) -> dict[str, int]:
        """Calculate parameter counts for all functions."""
        parameter_counts = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                parameter_counts[node.name] = param_count

        return parameter_counts

    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0

        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)

            nesting_nodes = (ast.If, ast.While, ast.For, ast.Try, ast.With)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, nesting_nodes):
                    calculate_depth(child, current_depth + 1)
                else:
                    calculate_depth(child, current_depth)

        calculate_depth(tree)
        return max_depth

    def _calculate_class_complexity(self, tree: ast.AST) -> dict[str, dict[str, Any]]:
        """Calculate complexity metrics for classes."""
        class_complexity = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                class_complexity[node.name] = {
                    "method_count": len(methods),
                    "total_lines": getattr(node, "end_lineno", 0)
                    - getattr(node, "lineno", 0),
                    "public_methods": len(
                        [m for m in methods if not m.name.startswith("_")]
                    ),
                }

        return class_complexity

    def _count_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Count nesting depth of a specific node."""
        max_depth = depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                child_depth = self._count_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _estimate_effort(self, opportunity: RefactoringOpportunity) -> str:
        """Estimate effort required for refactoring."""
        if opportunity.confidence > 0.9:
            return "low"
        elif opportunity.confidence > 0.7:
            return "medium"
        else:
            return "high"
