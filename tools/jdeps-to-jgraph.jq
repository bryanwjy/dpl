
# No copyright, written by ChatGPT
reduce .[] as $doc
  ({}; 
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
