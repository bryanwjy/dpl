#!/usr/bin/jq -f
# Written by Claude
($jmap[0]) as $jmap
| ($impl | split("\n") | map(select(length > 0))) as $impl
| . as $graph
| ($graph[$module] // error("module not found in graph: \($module)")) as $deps
| $deps # assume unique
| map($jmap[.] // error("module not found in jmap: \(.)"))
| map(sub("\\.pcm$"; ".o"))
| map(select(. as $p | $impl | index($p))) # assume unique
| .[]
