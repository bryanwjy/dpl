// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.algorithm;
export import :core.fwd;
import :core.details.algorithm;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)

namespace datapar {
inline namespace cpo {
using __DPL datapar::cpo::clamp;
using __DPL datapar::cpo::compress;
using __DPL datapar::cpo::expand;
using __DPL datapar::cpo::exscan;
using __DPL datapar::cpo::exscan_max;
using __DPL datapar::cpo::exscan_min;
using __DPL datapar::cpo::exscan_sum;
using __DPL datapar::cpo::hmax;
using __DPL datapar::cpo::hmin;
using __DPL datapar::cpo::hsum;
using __DPL datapar::cpo::inner_product;
using __DPL datapar::cpo::lerp;
using __DPL datapar::cpo::reduce;
using __DPL datapar::cpo::rotate_left;
using __DPL datapar::cpo::rotate_lefti;
using __DPL datapar::cpo::rotate_right;
using __DPL datapar::cpo::rotate_righti;
using __DPL datapar::cpo::sad;
using __DPL datapar::cpo::scan;
using __DPL datapar::cpo::scan_max;
using __DPL datapar::cpo::scan_min;
using __DPL datapar::cpo::scan_sum;
using __DPL datapar::cpo::shift_left;
using __DPL datapar::cpo::shift_lefti;
using __DPL datapar::cpo::shift_right;
using __DPL datapar::cpo::shift_righti;

} // namespace cpo
} // namespace datapar

// NOLINTEND(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_END
