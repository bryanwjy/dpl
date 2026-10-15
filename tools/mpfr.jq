#!/usr/bin/jq -f
# Written by Claude
# mpfr-mk.jq — generate make rules for every test's .mpfr.rsp
#
# Input (use -s): a stream of records, one per test, as written by the makefile:
#   {"file": "<abs path>/foo.pass.cpp.jdir", "directives": { ...directives.awk output... }}
#
# Arguments:
#   --arg bin '$(MPFR_BIN)'   text placed in the generated makefile for the generator
#                             binary (normally a make variable reference, kept unexpanded)
#
# Output: makefile text. For each test:
#   - with mpfr directives:
#       foo.pass.cpp.mpfr.rsp: foo.pass.cpp.jdir $(MPFR_BIN)
#               regenerate foo.pass.cpp.mpfr.dir by running each mpfr command in it,
#               then write "-I<dir>" to the .rsp
#   - without:
#       foo.pass.cpp.mpfr.rsp: foo.pass.cpp.jdir
#               remove any stale .mpfr.dir and write an empty .rsp

# Escape text for use inside a make recipe: "$" must be doubled.
def make_escape: gsub("\\$"; "$$");

# The global mpfr directive as a list of argument strings (accepts a single value or a list).
def mpfr_runs:
  (.global.mpfr // [])
  | if type == "array" then . else [.] end
  | map(tostring | select(length > 0));

.[]
| (.file | sub("\\.jdir$"; "")) as $base
| ($base + ".mpfr.rsp") as $rsp
| ($base + ".mpfr.dir") as $dir
| (.directives | mpfr_runs) as $runs
| if ($runs | length) == 0 then
    "\($rsp): \(.file)\n"
    + "\t@rm -rf '\($dir)'; : > '$@'\n"
  else
    "\($rsp): \(.file) \($bin)\n"
    + "\t@set -e; rm -rf '\($dir)'; mkdir -p '\($dir)'; cd '\($dir)'"
    + ($runs | map("; \($bin) \(make_escape)") | join(""))
    + "\n"
    + "\t@echo '-I\($dir)' > '$@'\n"
  end
