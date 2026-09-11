// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.immediate;
export import :core.fwd;
import :core.details.immediate;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)

namespace datapar {
using __DPL datapar::all_bits;
using __DPL datapar::all_bits_t;
using __DPL datapar::all_bits_v;
using __DPL datapar::broadcastable_base;
using __DPL datapar::const_mask;
using __DPL datapar::const_mask_like;
using __DPL datapar::deduce_const_mask_t;
using __DPL datapar::deduce_const_mask_v;
using __DPL datapar::imm;
using __DPL datapar::immediate;
using __DPL datapar::lsb;
using __DPL datapar::lsb_v;
using __DPL datapar::msb;
using __DPL datapar::msb_v;
using __DPL datapar::one;
using __DPL datapar::one_t;
using __DPL datapar::one_v;
using __DPL datapar::zero;
using __DPL datapar::zero_t;
using __DPL datapar::zero_v;
// TODO move and hide everything below under math
using __DPL datapar::infinity;
using __DPL datapar::infinity_v;
using __DPL datapar::max_value;
using __DPL datapar::max_value_v;
using __DPL datapar::min_value;
using __DPL datapar::min_value_v;
using __DPL datapar::nan;
using __DPL datapar::nan_v;
} // namespace datapar

// NOLINTEND(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_END
