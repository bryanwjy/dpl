
# Written by ChatGPT
reduce .[] as $doc
  ({}; 
   reduce $doc.rules[] as $rule
     (.;
      reduce $rule.provides[]? as $prov
        (.;
         .[$prov["logical-name"]] = $rule["primary-output"]
        )
     )
  )
