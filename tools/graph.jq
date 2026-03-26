#!/usr/bin/jq -f
# No copyright, written by Google Gemini
# Generate a lookup table for ALL modules at once
# 1. Build a Unified Graph
# This maps BOTH Logical Names and File Paths to their requirements
(reduce .rules[] as $item ({};
    ($item.provides[0]["logical-name"]) as $name
    | ($item["primary-output"]) as $path
    | ([$item.requires[]? | .["logical-name"]]) as $deps
    | . + (if $name then {($name): $deps} else {} end) # Map Name -> Deps
      + {($path): $deps}                               # Map Path -> Deps
)) as $graph

# 2. Compute Transitive Closure
# We iterate over the keys, but we only care about File Path keys for our final output
| $graph 
| with_entries(
    .key as $file
    | .value = (
        [ .value[] | recurse(. as $m | $graph[$m][]?) ] 
        | unique
      )
  )
