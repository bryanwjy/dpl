#!/usr/bin/jq -f
# Written by Claude
# mathdata.jq — turn every test's @dpl.mpfr directives into a makefile that
# generates one header per (function, type).
#
# Input (use -s): records written by the main makefile, one per test:
#   {"file": "<path>.jdir", "directives": { ...directives.awk output... }}
#
# Arguments:
#   --arg bin  <text>   generator reference placed in the output (e.g. '$(MPFR_BIN)')
#                       must be an absolute path once expanded
#
# Output: makefile text with
#   all: <every header>
#   <function>/<type>.h:   one rule per header (always regenerated)
#
# Paths are relative: run the result with `make -C <data dir> -f <this file>`,
# so headers are written under <data dir>.
#
# The same header may be requested by several tests; identical requests are merged.
# Different arguments for the same header are an error naming the tests involved.

def make_escape: gsub("\\$"; "$$");

# Fresh uint32 seed, evaluated by the shell when the rule runs ("$$" → "$" after make).
def seed: "$$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')";

def mpfr_runs:
  (.global.mpfr // [])
  | if type == "array" then . else [.] end
  | map(tostring | select(length > 0));

# One request: {out, args, file}
[ .[]
  | .file as $file
  | .directives | mpfr_runs[]
  | (split(" ") | map(select(length > 0))) as $a
  | if ($a | length) < 2 then
      error("\($file): mpfr directive needs a function and a type: '\(.)'")
    else
      {out: "\($a[0])/\($a[1]).h", args: ($a | join(" ")), file: $file}
    end
]
| group_by(.out)
| map(
    if (map(.args) | unique | length) > 1 then
      error("conflicting mpfr directives for \(.[0].out):\n"
            + (map("  \(.file): \(.args)") | join("\n")))
    else .[0] end
  )
| ( "all: " + (map(.out) | join(" ")) + "\n"
    + ".PHONY: all " + (map(.out) | join(" ")) + "\n\n"
    + (map("\(.out | make_escape):\n"
           + "\t@mkdir -p $(@D) && \($bin) \(.args | make_escape) --seed \(seed) -o $@\n"
           + "\t@echo \"MATHDATA $(CURDIR)/\(.out | make_escape)\"\n")
       | join("\n"))
  )
