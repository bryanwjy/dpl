# Copyright 2025-2026 Bryan Wong

from . import basic_simd

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        'type synthetic add -l lldb_formatter.basic_simd.BasicSimd.Synthetic -x "^dpl::datapar::basic_simd<.+>$"'
    )
    debugger.HandleCommand(
        'type summary add -F lldb_formatter.basic_simd.BasicSimd.summarize -x "^dpl::datapar::basic_simd<.+>$"'
    )
    debugger.HandleCommand(
        'type synthetic add -l lldb_formatter.basic_simd.BasicSimd.Synthetic -x "^basic_simd<.+>$"'
    )
    debugger.HandleCommand(
        'type summary add -F lldb_formatter.basic_simd.BasicSimd.summarize -x "^basic_simd<.+>$"'
    )
