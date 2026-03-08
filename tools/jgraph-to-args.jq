# tools/jgraph-to-args.jq

# unwrap slurpfiles
($jmap[0]) as $jmap
| ($jgraph[0]) as $jgraph

# extract target from jdep file itself
| (.rules[0].provides[0]?["logical-name"] // .rules[0]["primary-output"]) as $mod

# compute transitive closure using recursion
| ($mod | [recurse(. as $m | $jgraph[$m][]?)] ) 
# exclude self
| map(select(. != $mod))
# deduplicate and sort
| unique
# map to -fmodule-file
| map(
    if $jmap[.]? then
        "-fmodule-file=\(.)=\($jmap[.])"
    else
        error("unknown module: " + .)
    end
)
# emit one per line
| .[]
