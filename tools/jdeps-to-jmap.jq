
# Written by ChatGPT
(($ARGS.named.base // [{}])[0]) as $init
| reduce .[] as $doc
  ($init;
   reduce $doc.rules[] as $rule
     (.;
      reduce $rule.provides[]? as $prov
        (.;
         .[$prov["logical-name"]] = $rule["primary-output"]
        )
     )
  )
