/* Copyright 2023-2025 Bryan Wong */

#pragma once

/**
 * Concatenates the last token of the variadic argument with the first argument
 *
 * All of the arguments to a C preprocessor macro are fully expanded before the macro itself is
 * expanded, unless the # or ## operator is applied to them; then they're not expanded. So to get
 * full expansion before ##, pass the arguments through a wrapper macro that doesn't use ##.
 */
#define DPL_SWAP_CONCAT(Y, ...) DPL_PRIMITIVE_SWAP_CONCAT(Y, __VA_ARGS__)
/**
 * Internal implementation of DPL_SWAP_CONCAT @see DPL_SWAP_CONCAT
 */
#define DPL_PRIMITIVE_SWAP_CONCAT(Y, ...) __VA_ARGS__##Y

/**
 * Concatenates the first argument with the first token of the variadic argument
 */
#define DPL_CONCAT(_1, ...) DPL_PRIMITIVE_CONCAT(_1, __VA_ARGS__)
/**
 * Internal implementation of DPL_CONCAT @see DPL_CONCAT
 */
#define DPL_PRIMITIVE_CONCAT(_1, ...) _1##__VA_ARGS__
