# arguments:
#   $jmap = global logical-name → file map

def req_to_pcm($jmap):
  .["logical-name"] as $name
  | if $jmap[$name] then
      $jmap[$name]
    else
      error("unknown module: " + $name)
    end;

($jmap[0]) as $jmap
|
.rules[]
| .["primary-output"] as $out
| [
    .requires // [] | .[] | req_to_pcm($jmap)
  ] as $deps
| "\($out): \($deps | join(" "))"
