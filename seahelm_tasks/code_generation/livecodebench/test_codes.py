FUNCTION_TEST_CODE = """
import json
import sys
from typing import TYPE_CHECKING, Dict, List, Tuple, Set, Sequence, Mapping
import ast

#Convert multi-type string to list with original data type
def parse_mixed_data(data_string):
    lines = data_string.strip().split('\\n')
    result = []

    for line in lines:
        if line.strip():  # skip empty line
            try:
                parsed_value = ast.literal_eval(line.strip())
                result.append(parsed_value)
            except (ValueError, SyntaxError):
                result.append(line.strip()) # Keep as string if parse failed

    return result

# User's code
{code}

# Test execution for single test case
try:
    test_input = '''{test_input}'''
    expected_output = {expected_output}

    if 'class Solution' in '''{code}''':
        # LeetCode style
        solution = Solution()
        method = getattr(solution, '{fn_name}')
    else:
        # Function is directly available
        method = {fn_name}

    # Parse input if it's JSON string
    parse_multi_type = False
    if isinstance(test_input, str):
        try:
            if test_input.find("\\n") > -1:
                test_input = parse_mixed_data(test_input)
                parse_multi_type = True
            else:
                test_input = json.loads(test_input)
        except:
            pass  # Keep as string if not valid JSON

    # Call the method
    if parse_multi_type:
        result = method(*test_input)
    else:
        result = method(test_input)

    # Parse expected output if it's JSON string
    if isinstance(expected_output, str):
        try:
            expected_output = json.loads(expected_output)
        except:
            pass  # Keep as string if not valid JSON

    # Convert tuple to list for comparison
    if isinstance(result, tuple):
        result = list(result)

    if result == expected_output:
        print("TEST_PASSED")
    else:
        print(f"TEST_FAILED: expected {{expected_output}}, got {{result}}")

except Exception as e:
    print(f"EXECUTION_ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""

STDIN_TEST_CODE = """
import sys
from io import StringIO

# Redirect stdin
sys.stdin = StringIO('''{test_input}''')

# User's code
{code}
"""
