/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/preprocessor/is_empty.h"
#include "dpl/preprocessor/paste.h"

/**
 * TODO Use VA_OPT in C++20
 */

#define DPL_EXPAND_VA_ARGS_1()
#define DPL_EXPAND_VA_ARGS_0(...) , __VA_ARGS__
#define DPL_EXPAND_VA_ARGS(...) \
    DPL_PASTE(                  \
        DPL_APPEND_IS_EMPTY(DPL_EXPAND_VA_ARGS_, __VA_ARGS__)(__VA_ARGS__))
