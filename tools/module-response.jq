#!/usr/bin/jq -f
# Written by Google Gemini
($jmap[0]) as $m | .[$module][] | "-fmodule-file=\(.)=\($m[.])"
