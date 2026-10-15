
# Written by ChatGPT

# arguments:
#   $jmap = global logical-name → file map

def req_to_pcm($jmap; $whitelist):
  .["logical-name"] as $name
  | if $jmap | has($name) then
      $jmap[$name]
    elif any($whitelist[]; . == $name) then
      empty
    else
      error("unknown module: " + $name)
    end;

($jmap[0]) as $jmap
| ($ARGS.named.whitelist // []) as $whitelist
|
.rules[]
| .["primary-output"] as $out
| [
    .requires // [] | .[] | req_to_pcm($jmap; $whitelist)
  ] as $deps
| "\($out): \($deps | join(" "))"
