// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.xmm:math;
import :details.math;
import :basic;

// NOLINTBEGIN(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {
using __DPL datapar::xmm::fmadd;
// using __DPL datapar::xmm::fmaddsub;
// using __DPL datapar::xmm::fmsub;
// using __DPL datapar::xmm::fmsubadd;
// using __DPL datapar::xmm::fnmadd;
// using __DPL datapar::xmm::fnmsub;
} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END
// NOLINTEND(misc-unused-using-decls)
