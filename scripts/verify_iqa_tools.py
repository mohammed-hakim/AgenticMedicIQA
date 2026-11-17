#!/usr/bin/env python
"""
Verify IQA tool weight downloads and runtime availability.

Checks every registered IQA tool and confirms that:
1. IQA-PyTorch can create a tool instance.
2. Tool weights are available (downloading them on first use if needed).
3. The tool runs on a test image.
4. Score normalization produces a value in [1, 5].

The bundled tools were developed or calibrated on natural images. In Medical
AgenticIQA they are relative acquisition-quality proxies, not clinically
validated measurements.

Usage:
    python scripts/verify_iqa_tools.py                       # Test all tools
    python scripts/verify_iqa_tools.py --tools QAlign BRISQUE # Test named tools
    python scripts/verify_iqa_tools.py --type NR             # Test NR tools only
    python scripts/verify_iqa_tools.py --skip-slow           # Skip slow tools
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Add the project root to the import path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agentic.tool_registry import ToolRegistry, ToolExecutionError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ToolVerifier:
    """Verify IQA tool availability and execution."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.results: Dict[str, Dict] = {}

        # Tools that are known to take substantially longer to run.
        self.slow_tools = {'HyperIQA', 'MUSIQ', 'MANIQA', 'TReS'}

    def create_test_images(self, output_dir: Path) -> Tuple[Path, Path]:
        """
        Create a synthetic reference/distorted image pair.

        Args:
            output_dir: Directory for the generated images.

        Returns:
            Tuple of (reference_path, distorted_path).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        size = (256, 256)

        # Reference image: smooth color gradient.
        reference = Image.new('RGB', size)
        pixels = reference.load()
        for i in range(size[0]):
            for j in range(size[1]):
                pixels[i, j] = (
                    int(255 * i / size[0]),
                    int(255 * j / size[1]),
                    128,
                )

        reference_path = output_dir / 'reference.png'
        reference.save(reference_path)

        # Distorted image: additive Gaussian noise.
        distorted_array = np.array(reference)
        noise = np.random.normal(0, 25, distorted_array.shape)
        distorted_array = np.clip(distorted_array + noise, 0, 255).astype(np.uint8)
        distorted = Image.fromarray(distorted_array)

        distorted_path = output_dir / 'distorted.png'
        distorted.save(distorted_path)

        logger.info(f"Created test images: {reference_path}, {distorted_path}")
        return reference_path, distorted_path

    def verify_tool(
        self,
        tool_name: str,
        test_image: Path,
        reference_image: Optional[Path] = None
    ) -> Dict:
        """
        Verify a single IQA tool.

        Args:
            tool_name: Registered tool name.
            test_image: Path to the test image.
            reference_image: Path to the reference image (required by FR tools).

        Returns:
            Verification result dictionary.
        """
        result = {
            'tool': tool_name,
            'status': 'unknown',
            'error': None,
            'raw_score': None,
            'normalized_score': None,
            'execution_time': None,
        }

        try:
            if not self.registry.is_tool_available(tool_name):
                result['status'] = 'not_registered'
                result['error'] = f"Tool {tool_name} not found in registry"
                return result

            metadata = self.registry.tools[tool_name]
            tool_type = metadata['type']

            logger.info(f"Testing {tool_name} ({tool_type})...")

            if tool_type == 'FR' and reference_image is None:
                result['status'] = 'skipped'
                result['error'] = 'FR tool requires a reference image'
                return result

            start_time = time.time()

            try:
                if tool_type == 'FR':
                    raw_score, normalized_score = self.registry.execute_tool(
                        tool_name,
                        str(test_image),
                        reference_path=str(reference_image)
                    )
                else:
                    raw_score, normalized_score = self.registry.execute_tool(
                        tool_name,
                        str(test_image)
                    )

                execution_time = time.time() - start_time

                if not np.isfinite(raw_score):
                    result['status'] = 'invalid_score'
                    result['error'] = f'Non-finite raw score: {raw_score}'
                    return result

                if not (1.0 <= normalized_score <= 5.0):
                    result['status'] = 'invalid_normalized'
                    result['error'] = (
                        f'Normalized score {normalized_score} out of range [1, 5]'
                    )
                    return result

                result['status'] = 'success'
                result['raw_score'] = float(raw_score)
                result['normalized_score'] = float(normalized_score)
                result['execution_time'] = execution_time

                logger.info(
                    f"  OK {tool_name}: raw={raw_score:.4f}, "
                    f"normalized={normalized_score:.2f}, time={execution_time:.2f}s"
                )

            except ImportError as e:
                result['status'] = 'missing_dependency'
                result['error'] = f'Missing dependency: {str(e)}'
                logger.error(f"  FAIL {tool_name}: {result['error']}")

            except ToolExecutionError as e:
                result['status'] = 'execution_error'
                result['error'] = str(e)
                logger.error(f"  FAIL {tool_name}: {result['error']}")

            except Exception as e:
                result['status'] = 'unknown_error'
                result['error'] = f'{type(e).__name__}: {str(e)}'
                logger.error(f"  FAIL {tool_name}: {result['error']}")

        except Exception as e:
            result['status'] = 'verification_error'
            result['error'] = f'Verification failed: {str(e)}'
            logger.error(f"  FAIL {tool_name}: {result['error']}")

        return result

    def verify_all_tools(
        self,
        tool_names: Optional[List[str]] = None,
        tool_type: Optional[str] = None,
        skip_slow: bool = False
    ) -> Dict[str, Dict]:
        """
        Verify multiple tools.

        Args:
            tool_names: Tools to test (None means all registered tools).
            tool_type: Optional type filter ("FR" or "NR").
            skip_slow: Skip tools in the known-slow set.

        Returns:
            Mapping of tool name to result dictionary.
        """
        test_dir = Path(tempfile.mkdtemp(prefix='iqa_tool_test_'))
        reference_path, distorted_path = self.create_test_images(test_dir)

        if tool_names is None:
            all_tools = [
                name for name in self.registry.tools.keys()
                if not name.startswith('_')
            ]
        else:
            all_tools = tool_names

        if tool_type:
            all_tools = [
                name for name in all_tools
                if self.registry.tools.get(name, {}).get('type') == tool_type
            ]

        if skip_slow:
            all_tools = [
                name for name in all_tools
                if name not in self.slow_tools
            ]

        logger.info(f"\nVerifying {len(all_tools)} tools...\n")

        results = {}
        for i, tool_name in enumerate(all_tools, 1):
            logger.info(f"[{i}/{len(all_tools)}] {tool_name}")
            results[tool_name] = self.verify_tool(
                tool_name,
                test_image=distorted_path,
                reference_image=reference_path
            )

        try:
            reference_path.unlink()
            distorted_path.unlink()
            test_dir.rmdir()
        except Exception:
            pass

        return results

    def print_summary(self, results: Dict[str, Dict]) -> int:
        """Print a verification summary and return an exit code."""
        total = len(results)
        success = sum(1 for r in results.values() if r['status'] == 'success')
        failed = total - success

        by_status: Dict[str, List[str]] = {}
        for tool_name, result in results.items():
            by_status.setdefault(result['status'], []).append(tool_name)

        print("\n" + "=" * 70)
        print("Verification summary")
        print("=" * 70)
        print(f"Total: {total} tools")
        if total:
            print(f"Passed: {success} tools ({success / total * 100:.1f}%)")
            print(f"Failed: {failed} tools ({failed / total * 100:.1f}%)")
        print()

        for status, tools in sorted(by_status.items()):
            if status == 'success':
                print(f"PASS ({len(tools)}):")
                for tool in sorted(tools):
                    r = results[tool]
                    print(
                        f"  - {tool}: normalized={r['normalized_score']:.2f}, "
                        f"time={r['execution_time']:.2f}s"
                    )
            else:
                print(f"FAIL - {status.replace('_', ' ').title()} ({len(tools)}):")
                for tool in sorted(tools):
                    error_msg = results[tool].get('error', 'Unknown error')
                    if len(error_msg) > 100:
                        error_msg = error_msg[:97] + "..."
                    print(f"  - {tool}: {error_msg}")
            print()

        if 'success' in by_status:
            execution_times = [
                (tool, results[tool]['execution_time'])
                for tool in by_status['success']
            ]
            execution_times.sort(key=lambda item: item[1], reverse=True)

            print("Execution time ranking (top 5 slowest tools):")
            for tool, exec_time in execution_times[:5]:
                print(f"  - {tool}: {exec_time:.2f}s")
            print()

        print("=" * 70)

        return 0 if failed == 0 else 1


def main() -> int:
    """Run the tool verifier CLI."""
    parser = argparse.ArgumentParser(
        description=(
            'Verify IQA tool weight downloads and runtime availability. '
            'Scores are uncalibrated natural-image proxies, not clinical measurements.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all tools
  python scripts/verify_iqa_tools.py

  # Test named tools
  python scripts/verify_iqa_tools.py --tools QAlign BRISQUE TOPIQ_FR

  # Test NR tools only
  python scripts/verify_iqa_tools.py --type NR

  # Skip slow tools
  python scripts/verify_iqa_tools.py --skip-slow

  # Combine filters
  python scripts/verify_iqa_tools.py --type FR --skip-slow
        """
    )

    parser.add_argument(
        '--tools',
        nargs='+',
        help='Tool names to test (default: all registered tools)'
    )

    parser.add_argument(
        '--type',
        choices=['FR', 'NR'],
        help='Only test this tool type (FR: Full-Reference, NR: No-Reference)'
    )

    parser.add_argument(
        '--skip-slow',
        action='store_true',
        help='Skip known slow tools (HyperIQA, MUSIQ, MANIQA, TReS)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    try:
        logger.info("Loading tool registry...")
        registry = ToolRegistry()
        total_tools = len([k for k in registry.tools.keys() if not k.startswith('_')])
        logger.info(f"Loaded {total_tools} tools")
    except Exception as e:
        logger.error(f"Unable to load tool registry: {e}")
        return 1

    verifier = ToolVerifier(registry)

    try:
        results = verifier.verify_all_tools(
            tool_names=args.tools,
            tool_type=args.type,
            skip_slow=args.skip_slow
        )
        return verifier.print_summary(results)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
