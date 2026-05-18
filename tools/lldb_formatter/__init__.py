# Copyright 2025-2026 Bryan Wong

from . import classes

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        'type synthetic add -w dpl -l lldb_formatter.classes.FormatDispatcher -x "^basic_vector"'
    )
    debugger.HandleCommand(
        'type summary add -w dpl -F lldb_formatter.classes.summarize -x "^basic_vector"'
    )
    debugger.HandleCommand(
        'type synthetic add -w dpl -l lldb_formatter.classes.FormatDispatcher -x "^dpl::datapar::basic_vector"'
    )
    debugger.HandleCommand(
        'type summary add -w dpl -F lldb_formatter.classes.summarize -x "^dpl::datapar::basic_vector"'
    )
    debugger.HandleCommand(
        'command script add -f lldb_formatter.classes.bf16.print pbf16'
    )
    debugger.HandleCommand(
        'type category enable dpl'
    )
