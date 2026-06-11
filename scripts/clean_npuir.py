#!/usr/bin/env python3
"""Clean NPUIR file: strip bishengir-compile warnings and incompatible ops.

The remote 910B3 bishengir may emit ops (e.g. hivm.hir.copy with ub->ub
address spaces) whose format is incompatible with the local MLIR parser's
HIVM dialect definition.  This script strips:
  - Compiler/linker warning lines (loc(...): warning:, ld.lld:, etc.)
  - hivm.hir.copy ops that fail local dialect parsing
"""
import sys

src = sys.argv[1]
dst = sys.argv[2]

with open(src) as f:
    lines = f.readlines()

def is_mlir_line(line):
    s = line.lstrip()
    # Strip compiler/linker warnings
    if s.startswith('loc(') and 'warning' in s:
        return False
    if s.startswith(('ld.lld:', '[ERROR]', '[WARNING]', '[INFO]')):
        return False
    if s.startswith('warning') or s.endswith('warning generated.\n'):
        return False
    # Strip hivm.hir.copy ops (ub->ub copies fail local parser)
    if 'hivm.hir.copy' in s:
        return False
    # Allowlist: valid MLIR content
    if not s:  # blank line
        return True
    if s.startswith(('//', 'func', '%', 'hivm', 'arith', 'memref',
                     'scf', 'return', 'module', '#', '{', '}',
                     'annotation', 'affine', 'cf', 'test', 'linalg',
                     'builtin', 'vector', 'math', 'ub', 'bufferization',
                     'tensor', 'transform')):
        return True
    return False

clean = [l for l in lines if is_mlir_line(l)]

with open(dst, 'w') as f:
    f.writelines(clean)

print(f"Lines: {len(lines)} -> {len(clean)} (stripped {len(lines)-len(clean)})")
print(f"First 3 clean lines:")
for l in clean[:3]:
    print(l.rstrip())
