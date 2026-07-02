#!/usr/bin/jq -f
# Written by Claude
($jmap[0]) as $jmap
| ($impl | split("\n") | map(select(length > 0))) as $impl
| . as $graph
| [ $module | recurse($graph[.][]?) ]
| unique
| map($jmap[.] // empty)
| map(sub("\\.pcm$"; ".o"))
| map(select(. as $p | $impl | index($p)))
| unique
| .[]
