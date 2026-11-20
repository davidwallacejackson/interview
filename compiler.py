#!/usr/bin/env python3
"""Luminal Compiler Take Home — compiler engineering candidate implementation.

This implementation optimizes scratch use by reclaiming scratch space after a
variable's final read. It stores only one type of data (scalar or vector) in
a given memory location (mixed reuse would be a future optimization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import sys
import typing as t

import machine

type Operation = dict
type OperationId = int


class VariableType(Enum):
    SCALAR = 0
    VECTOR = 1

    @staticmethod
    def from_op_result(op: Operation) -> VariableType:
        as_str = machine.OP_SPECS[op["op"]]["result"]
        return VariableType.SCALAR if as_str == "scalar" else VariableType.VECTOR


@dataclass
class Variable:
    operations: list[Operation]

    written_by_id: t.Optional[OperationId] = None
    read_by_ids: t.Set[OperationId] = field(default_factory=set)
    var_type: t.Optional[VariableType] = None

    # 
    mem_slot: t.Optional[int] = None


    @staticmethod
    def factory_for_ops(operations: list[Operation]):
        return lambda: Variable(operations)

    def written_by(self) -> Operation:
        if self.written_by_id is None:
            raise Exception("called written_by() before setting writer")

        return self.operations[self.written_by_id]

    def read_by(self) -> t.Set[Operation]:
        return set([self.operations[id] for id in self.read_by_ids])


def get_vars(operations: list[Operation]):
    errs: list[str] = []
    vars: defaultdict[str, Variable] = defaultdict(
        Variable.factory_for_ops(operations)
    )

    # build a graph of variable assignments and reads:
    for op in operations:
        for arg in op.get("args", []):
            if vars[arg].written_by_id is None:
                errs.append(f"read {arg} before write at instruction {op["id"]} ({op["op"]})")
            vars[arg].read_by_ids.add(op["id"])

        if "dest" in op:
            dest = op["dest"]
            if vars[dest].written_by_id is not None:
                errs.append(f"rewrote {dest} at instruction {op["id"]} ({op["op"]})")

            vars[dest].written_by_id = op["id"]
            vars[dest].var_type = VariableType.from_op_result(op)

    if errs:
        raise Exception("errors getting variable graph", errs)

    return vars


def compile_program(program: dict) -> dict:
    """Compile one validated IR program into scratch allocations and bundles."""

    # A simple non-overlapping allocation. Vectors are placed first so their
    # alignment does not create holes between scalar values.
    scratch: dict[str, int] = {}
    cursor = 0
    operations: list[Operation] = program["operations"]

    vars = get_vars(operations)

    scalar_cursor = 0
    vector_cursor = 0

    # pools of free memory locations that have opened up for reuse
    # (expressed in terms of var type-sized "slots" rather than absolute
    # memory locations)

    scalar_pool: list[int] = []
    vector_pool: list[int] = []

    for op in operations:
        id = op["id"]

        # free any vars we read this op, if this was their last read
        for arg in op.get("args", []):
            var = vars[arg]
            if max(var.read_by_ids) == id:
                if var.var_type == VariableType.SCALAR:
                    scalar_pool.append(t.cast(int, var.mem_slot))
                else:
                    vector_pool.append(t.cast(int, var.mem_slot))

        # pick a spot for this operation's result, if any:
        if dest := op.get("dest"):
            var = vars[dest]
            if var.var_type == VariableType.SCALAR:
                if len(scalar_pool) > 0:
                    var.mem_slot = scalar_pool.pop()
                    continue

                var.mem_slot = scalar_cursor
                scalar_cursor += 1
            else:
                if len(vector_pool) > 0:
                    var.mem_slot = vector_pool.pop()
                    continue

                var.mem_slot = vector_cursor
                vector_cursor += 1

    for result_kind in ("vector", "scalar"):
        for operation in operations:
            spec = machine.OP_SPECS[operation["op"]]
            if spec["result"] != result_kind:
                continue
            dest = operation["dest"]
            if result_kind == "vector":
                cursor = machine.align_up(cursor, machine.VLEN)
                scratch[dest] = cursor
                cursor += machine.VLEN
            else:
                scratch[dest] = cursor
                cursor += 1

    if cursor > machine.SCRATCH_WORDS:
        raise machine.CompileError(
            f"program requires {cursor} scratch words, limit is {machine.SCRATCH_WORDS}"
        )

    # Serial, source-order scheduling with explicit latency stalls. This is a
    # correct baseline, but it leaves almost all VLIW slots empty.
    bundles: list[dict[str, list[int]]] = []
    issue_cycle: dict[int, int] = {}
    producer = machine.producer_map(program)

    for operation in operations:
        spec = machine.OP_SPECS[operation["op"]]
        earliest = len(bundles)

        for arg in operation.get("args", []):
            pred_id = producer[arg]
            pred = operations[pred_id]
            earliest = max(
                earliest,
                issue_cycle[pred_id] + machine.OP_SPECS[pred["op"]]["latency"],
            )

        for pred_id in machine.memory_predecessors(program, operation["id"]):
            earliest = max(earliest, issue_cycle[pred_id] + 1)

        while len(bundles) < earliest:
            bundles.append({})

        bundles.append({spec["engine"]: [operation["id"]]})
        issue_cycle[operation["id"]] = earliest

    return {"scratch": scratch, "bundles": bundles}



def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python3 compiler.py <program.json>", file=sys.stderr)
        return 2

    program = machine.load_program(argv[0])
    compilation = compile_program(program)
    machine.check_compilation(program, compilation)
    json.dump(compilation, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


