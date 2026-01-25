reduce .[] as $doc
  ({}; 
   reduce ($doc.rules[]?) as $rule
     (.;
      ($rule.provides[]?["logical-name"]) as $prov
      |
      .[$prov] =
        ($rule.requires // []
         | map(.["logical-name"]))
     )
  )
