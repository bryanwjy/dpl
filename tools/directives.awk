#!/usr/bin/awk -f
# Written by Claude
#
# Extracts @dpl directives from a file's leading comment prologue and prints JSON.
# Usage: awk -v family=<name> -f directives.awk file.cpp
#
# Directive:   @dpl.<key>[.<subkey>...]: value             (global)
#              @dpl[<family>].<key>[.<subkey>...]: value   (family-specific)
#   - Key segments: [A-Za-z0-9_-]+
#   - A trailing "\" continues the value onto the next line (joined by one space).
#   - "key: v" sets a single string value: "key": "v". Setting it twice is an error.
#   - Dotted keys nest: "a.b.c: v" yields {"a": {"b": {"c": "v"}}}.
#   - "+" suffix = append to a list:
#       "key+: v"      appends string v:            "key": ["v", ...]
#       "key+.sub: v"  appends a new object:        "key": [{"sub": "v"}, ...]
#     The same segment without "+" refers to the most recent object element:
#         @dpl.run+.type: f32          -> "run": [ {"type": "f32", "range": "-1:1"},
#         @dpl.run.range: -1:1                     {"type": "f64"} ]
#         @dpl.run+.type: f64
#     Lists nest ("a+.b+.c"). A list holds strings or objects, not both.
#     Forgetting "+" on a later object element merges it into the previous one
#     (an error only if it re-sets a key that element already has).
#   - A comment line containing "@dpl." or "@dpl[" that does not match the
#     directive syntax is an error (e.g. "range.+type" instead of "range+.type").
#   - Using one path as two kinds of node (value, object or list; e.g. "key"
#     and "key.sub" in the same namespace) is an error: exit 1 with a message.
#
# Families: a family is any label, not necessarily a compiler. Output has a
# "global" object plus, if -v family=X is given, an "X" object holding the
# @dpl[X] directives; other families are dropped. The two objects are
# independent: the same key may appear in both, with different shapes. Merging
# them (e.g. family overriding global) is left to the consumer.

BEGIN {
    in_block = 0
    in_cont = 0
    ns_count = 0
    failed = 0
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
    if (match(clean_line, /@dpl(\[[^]]+\])?\.[a-zA-Z0-9_-]+\+?(\.[a-zA-Z0-9_-]+\+?)*:[ \t]*/)) {
        header = substr(clean_line, RSTART, RLENGTH)
        content = substr(clean_line, RSTART + RLENGTH)

        ns = "global"
        if (header ~ /\[/) {
            ns = header
            sub(/^@dpl\[/, "", ns); sub(/\].*/, "", ns)
        }

        key = header
        sub(/^@dpl(\[[^]]+\])?\./, "", key); sub(/:[ \t]*$/, "", key)

        if (sub(/[ \t]*\\$/, "", content)) {
            in_cont = 1
            val = trim(content)
        } else {
            store_directive(ns, key, content)
        }
    } else if (clean_line ~ /@dpl[.[]/) {
        die("malformed directive: " trim(clean_line))
    }
    if (block_ended_here) in_block = 0
}

END {
    if (failed) exit 1
    print_json()
}

# --- FUNCTIONS ---

function trim(s) {
    gsub(/^[ \t\r\n]+/, "", s); gsub(/[ \t\r\n]+$/, "", s)
    return s
}

function die(msg) {
    printf "%s:%d: error: %s\n", FILENAME, FNR, msg > "/dev/stderr"
    failed = 1
    exit 1
}

# Tree per namespace n, keyed by internal path. Paths join segments with ".";
# list elements are "<list path>#<index>" ("#" cannot appear in a key).
#   is_obj[n, path]          path is an object (object-list elements too)
#   is_list[n, path]         path is a list: "obj" (of objects) or "str" (of values)
#   elem_count[n, path]      number of elements in list path (last = current)
#   is_val[n, path]          path is a single value
#   vals[n, path]            the value of a single value or "str"-list element
#   kids[n, path, i]         i-th child segment of object path ("" = root)
function store_directive(n, k, v,   segs, m, j, seg, plus, last, p, child, kind) {
    if (n != "global" && n != family) return

    if (!(n in ns_seen)) {
        ns_list[++ns_count] = n
        ns_seen[n] = 1
    }

    m = split(k, segs, ".")
    p = ""
    for (j = 1; j <= m; j++) {
        seg = segs[j]
        plus = sub(/\+$/, "", seg)
        last = (j == m)
        child = (p == "" ? seg : p "." seg)

        if (plus) {
            if ((n, child) in is_obj || (n, child) in is_val)
                die("'" k "': '" seg "' was already used as a non-list")
            kind = (last ? "str" : "obj")
            if (!((n, child) in is_list)) {
                is_list[n, child] = kind
                kids[n, p, ++kid_count[n, p]] = seg
            } else if (is_list[n, child] != kind) {
                die("'" k "': list '" seg "' cannot mix values and objects")
            }
            p = child "#" (++elem_count[n, child])
            if (last) vals[n, p] = trim(v)
            else is_obj[n, p] = 1
        } else if ((n, child) in is_list) {
            if (last) die("'" k "': '" seg "' is a list; use '" seg "+' to append")
            if (is_list[n, child] == "str") die("'" k "': '" seg "' is a list of values; it cannot have subkeys")
            p = child "#" elem_count[n, child]
        } else if (last) {
            if ((n, child) in is_obj) die("'" k "': '" seg "' already has subkeys; it cannot also hold a value")
            if ((n, child) in is_val) die("'" k "': '" seg "' is already set; use '" seg "+' for a list")
            is_val[n, child] = 1
            kids[n, p, ++kid_count[n, p]] = seg
            vals[n, child] = trim(v)
        } else {
            if ((n, child) in is_val) die("'" k "': '" seg "' already holds a value; it cannot also have subkeys")
            if (!((n, child) in is_obj)) {
                is_obj[n, child] = 1
                kids[n, p, ++kid_count[n, p]] = seg
            }
            p = child
        }
    }
}

function escape(s) {
    gsub(/\\/, "\\\\", s); gsub(/"/, "\\\"", s)
    return s
}

function print_obj(n, p, ind,   i, c, path, m, cnt) {
    for (i = 1; i <= kid_count[n, p]; i++) {
        c = kids[n, p, i]
        path = (p == "" ? c : p "." c)
        printf "%s\"%s\": ", ind, escape(c)
        if ((n, path) in is_list) {
            print "["
            cnt = elem_count[n, path]
            for (m = 1; m <= cnt; m++) {
                if (is_list[n, path] == "str") {
                    printf "%s  \"%s\"", ind, escape(vals[n, path "#" m])
                } else {
                    printf "%s  {\n", ind
                    print_obj(n, path "#" m, ind "    ")
                    printf "%s  }", ind
                }
                print (m == cnt ? "" : ",")
            }
            printf "%s]", ind
        } else if ((n, path) in is_obj) {
            print "{"
            print_obj(n, path, ind "  ")
            printf "%s}", ind
        } else {
            printf "\"%s\"", escape(vals[n, path])
        }
        print (i == kid_count[n, p] ? "" : ",")
    }
}

function print_json(   i, n) {
    print "{"
    for (i = 1; i <= ns_count; i++) {
        n = ns_list[i]
        printf "  \"%s\": {\n", escape(n)
        print_obj(n, "", "    ")
        printf "  }%s\n", (i == ns_count ? "" : ",")
    }
    print "}"
}
