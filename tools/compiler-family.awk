#!/usr/bin/awk -f
# No copyright; written by Google Gemini

/#define[ \t]+DPL_COMPILER_[A-Z_]+[ \t]+1/ {
    # Match the specific part we want: DPL_COMPILER_...
    if (match($0, /DPL_COMPILER_[A-Z_]+/)) {
        # Extract the matched string
        str = substr($0, RSTART, RLENGTH)
        # Remove the prefix "DPL_COMPILER_"
        sub(/^DPL_COMPILER_/, "", str)
        print tolower(str)
        exit
    }
}
