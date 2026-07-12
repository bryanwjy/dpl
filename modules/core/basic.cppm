// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.basic;
export import :core.fwd;
import :core.details.basic;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)

namespace datapar {
using __DPL datapar::aligned;
using __DPL datapar::aligned_t;
using __DPL datapar::basic_mask;
using __DPL datapar::basic_vector;
using __DPL datapar::broadcasting;
using __DPL datapar::broadcasting_t;
using __DPL datapar::to_canonical;
using __DPL datapar::to_native_type;
//

inline namespace cpo {
using __DPL datapar::cpo::aligned_load;
using __DPL datapar::cpo::aligned_store;
using __DPL datapar::cpo::broadcast;
using __DPL datapar::cpo::expand_load;
using __DPL datapar::cpo::extract;
using __DPL datapar::cpo::from_bitset;
using __DPL datapar::cpo::gather;
using __DPL datapar::cpo::initialize;
using __DPL datapar::cpo::lane_index;
using __DPL datapar::cpo::load;
using __DPL datapar::cpo::store;
using __DPL datapar::cpo::to_bitset;
} // namespace cpo
} // namespace datapar

// NOLINTEND(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_END
