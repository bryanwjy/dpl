// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:std.utility;
export import :std.stddef;
import :std.details.utility;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)
inline namespace bit_literals {
using __DPL bit_literals::operator""_bits;
}

using __DPL bit_width;
using __DPL bitset;
using __DPL bitset_constant_like;
using __DPL bitset_type;
using __DPL countl_one;
using __DPL countl_zero;
using __DPL countr_one;
using __DPL countr_zero;
using __DPL get;
using __DPL integral_bitset_type;
using __DPL popcount;
using __DPL rotl;
using __DPL rotr;
// using __DPL to_underlying; // Exported below
using __DPL truncate;
//
using __DPL apply;
using __DPL as_const;
using __DPL exchange;
using __DPL forward;
using __DPL forward_like;
using __DPL ignore;
using __DPL ignore_t;
using __DPL move;
using __DPL move_if_noexcept;
using __DPL template_barrier;
using __DPL template_barrier_t;
using __DPL to_signed;
using __DPL to_underlying;
using __DPL to_unsigned;
using __DPL unreachable;
//
using __DPL constant_type_pack;
using __DPL type_pack;
using __DPL value_pack;
namespace pack {
using __DPL pack::all_of;
using __DPL pack::any_of;
using __DPL pack::for_each;
using __DPL pack::none_of;
} // namespace pack
// NOLINTEND(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_END
