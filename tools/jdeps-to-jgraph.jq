
# Written by ChatGPT
(($ARGS.named.base // {}) | if type == "array" then .[0] else . end) as $init
| reduce .[] as $doc
  ($init;
   reduce ($doc.rules[]?) as $rule
     (.;
      (($rule.provides // [{"logical-name": $rule["primary-output"]}])
       []["logical-name"]) as $prov
      |
      .[$prov] =
        ($rule.requires // []
         | map(.["logical-name"]))
     )
  )
