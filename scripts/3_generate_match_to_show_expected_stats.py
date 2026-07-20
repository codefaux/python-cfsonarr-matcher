#!/usr/bin/env python3

# - Goal:
#  Generate tests/fixtures/match_to_show/expected_stats.repr from tests/fixtures/initial/match_to_show/*
#  Sanity checks?  Detect conflicting decisions etc, alert user

# Enumerate ./tests/fixtures/initial/match_to_show/func_*.reprl
# Unique filter
# Sanity checks
# Write ./tests/fixtures/match_to_show/expected_stats.reprl

import ast
import hashlib
import marshal
from pathlib import Path

from cfsonarrmatcher import match_to_show

co_code_hash = hashlib.sha256(
    marshal.dumps(match_to_show.__code__.co_code)
).hexdigest()[:24]

input_file = Path(
    Path.cwd()
    / f"tests/fixtures/match_to_show/initial/match_to_show/func_{co_code_hash}.reprl"
)
output_file = Path(Path.cwd() / "tests/fixtures/match_to_show/expected_stats.reprl")

output_data: list[dict[str, str]] = []

with input_file.open("r", encoding="utf-8") as f:
    for line in f:
        obj = ast.literal_eval(line)
        output_data.append(obj)

if output_file.exists():
    output_file.unlink()

with output_file.open("w", encoding="utf-8") as f:
    f.write(repr(output_data))
    f.write("\n")

print(f"-- {output_file.relative_to(Path.cwd())} updated.")
