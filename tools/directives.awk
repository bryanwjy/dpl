#!/usr/bin/awk -f
# No copyright; written by Google Gemini

BEGIN {
    in_block = 0
    in_cont = 0
    ns_count = 0
}

{
    line = $0
    clean_line = line

    # 1. Identify Comment Context
    is_line_comment = 0
    if (!in_block) {
        if (match(clean_line, /^[ \t]*\/\//)) {
            is_line_comment = 1
            clean_line = substr(clean_line, RSTART + RLENGTH)
        } else if (match(clean_line, /^[ \t]*\/\*/)) {
            in_block = 1
            clean_line = substr(clean_line, RSTART + RLENGTH)
        }
    }
    
    block_ended_here = 0
    if (in_block && match(clean_line, /\*\//)) {
        block_ended_here = 1
        clean_line = substr(clean_line, 1, RSTART - 1)
    }

    # 2. Prologue Termination
    if (!in_block && !is_line_comment && !in_cont) {
        test_line = line
        gsub(/^[ \t]+/, "", test_line)
        if (test_line != "") exit 
    }

    # 3. Handle Continuation
    if (in_cont) {
        if (sub(/[ \t]*\\$/, "", clean_line)) {
            val = val " " trim(clean_line)
        } else {
            val = val " " trim(clean_line)
            store_directive(ns, key, val)
            in_cont = 0
        }
        if (block_ended_here) in_block = 0
        next
    }

    # 4. Directive Extraction
    if (match(clean_line, /@dpl(\[[^]]+\])?\.[a-zA-Z0-9_-]+:[ \t]*/)) {
        header = substr(clean_line, RSTART, RLENGTH)
        content = substr(clean_line, RSTART + RLENGTH)

        ns = "global"
        if (header ~ /\[/) {
            ns = header
            sub(/^@dpl\[/, "", ns); sub(/\].*/, "", ns)
        }

        key = header
        sub(/.*[.]/, "", key); sub(/:[ \t]*$/, "", key)

        if (sub(/[ \t]*\\$/, "", content)) {
            in_cont = 1
            val = trim(content)
        } else {
            store_directive(ns, key, content)
        }
    }
    if (block_ended_here) in_block = 0
}

END { print_json() }

# --- FUNCTIONS ---

function trim(s) {
    gsub(/^[ \t\r\n]+/, "", s); gsub(/[ \t\r\n]+$/, "", s)
    return s
}

# Add 'family' to your awk variables (passed via -v family=...)
function store_directive(n, k, v) {
    # Only store if it's 'global' or matches the current compiler family
    if (n != "global" && n != family) {
        return
    }

    if (!(n in ns_seen)) {
        ns_list[++ns_count] = n
        ns_seen[n] = 1
    }
    
    if (k == "compile-flags") {
        idx = ++k_array_count[n, k]
        vals[n, k, idx] = trim(v)
        if (!key_seen[n, k]++) key_list[n, ++key_count[n]] = k
    } else {
        if (!key_seen[n, k]++) key_list[n, ++key_count[n]] = k
        vals[n, k, 1] = trim(v)
    }
}

function escape(s) {
    gsub(/\\/, "\\\\", s); gsub(/"/, "\\\"", s)
    return s
}

function print_json(   i, j, m, n, k, ki) {
    print "{"
    for (i = 1; i <= ns_count; i++) {
        n = ns_list[i]
        printf "  \"%s\": {\n", escape(n)
        for (j = 1; j <= key_count[n]; j++) {
            k = key_list[n, j]
            printf "    \"%s\": ", escape(k)
            
            if (k == "compile-flags") {
                printf "[\n"
                for (m = 1; m <= k_array_count[n, k]; m++) {
                    printf "      \"%s\"%s\n", escape(vals[n, k, m]), (m == k_array_count[n, k] ? "" : ",")
                }
                printf "    ]%s\n", (j == key_count[n] ? "" : ",")
            } else {
                printf "\"%s\"%s\n", escape(vals[n, k, 1]), (j == key_count[n] ? "" : ",")
            }
        }
        printf "  }%s\n", (i == ns_count ? "" : ",")
    }
    print "}"
}
