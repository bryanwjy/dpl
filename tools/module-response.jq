#!/usr/bin/jq -f
# No copyright, written by Google Gemini
($jmap[0]) as $m | .[$module][] | "-fmodule-file=\(.)=\($m[.])"
