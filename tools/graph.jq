#!/usr/bin/jq -f
# Written by Claude
# Generate a lookup table for ALL modules at once
# 1. Build the graph, keyed once by logical-name only (no duplicate path entries)
(reduce .rules[] as $item (
    {by_name: {}, path_only: {}};
    ($item.provides[0]["logical-name"]) as $name
    | ($item["primary-output"]) as $path
    | ([$item.requires[]? | .["logical-name"]]) as $deps
    | .by_name += (if $name then {($name): $deps} else {} end)
    | .path_only += (if $path then {($path): {name: $name, deps: $deps}} else {} end)
  )
) as $idx

| $idx.by_name as $graph

# 2. Memoized transitive closure: each node computed exactly once,
#    reusing already-computed dependency closures instead of recursing from scratch.
| def closure($node; $memo):
    if ($memo | has($node)) then $memo
    else
      ($graph[$node] // []) as $deps
      | (reduce $deps[] as $d ($memo; closure($d; .))) as $memo2
      | ( $deps + [ $deps[] as $d | $memo2[$d][]? ] | unique ) as $result
      | $memo2 + {($node): $result}
    end;

(reduce ($graph | keys_unsorted[]) as $k ({}; closure($k; .))) as $by_name

| ($idx.path_only
    | with_entries(
        .value = (
          if .value.name then $by_name[.value.name]
          else
            # path-only rule (no provided module): compute its own closure
            # directly, still reusing memoized by-name closures for its deps
            (.value.deps) as $deps
            | ( $deps + [ $deps[] as $d | $by_name[$d][]? ] | unique )
          end
        )
      )
  ) as $by_path

| $by_name + $by_path
