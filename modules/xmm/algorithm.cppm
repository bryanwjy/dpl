// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.xmm:algorithm;
import :details.algorithm;
import :basic;

// NOLINTBEGIN(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {
using __DPL datapar::xmm::slide_left;
using __DPL datapar::xmm::slide_right;
} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END
// NOLINTEND(misc-unused-using-decls)
