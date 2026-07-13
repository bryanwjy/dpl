// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.xmm:basic;
import :details.basic;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)
namespace datapar::xmm {
using __DPL datapar::xmm::abi;
using __DPL datapar::xmm::abi_tag;
using __DPL datapar::xmm::broadcast;
using __DPL datapar::xmm::extract;
using __DPL datapar::xmm::from_bitset;
using __DPL datapar::xmm::gather;
using __DPL datapar::xmm::initialize;
using __DPL datapar::xmm::load;
using __DPL datapar::xmm::mask;
using __DPL datapar::xmm::store;
using __DPL datapar::xmm::to_bitset;
using __DPL datapar::xmm::vector;
} // namespace datapar::xmm
namespace xmm = datapar::xmm;
// NOLINTEND(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_END
