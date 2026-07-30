// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.utility;
import :core.details.utility;

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
// NOLINTBEGIN(misc-unused-using-decls)
using __DPL datapar::swap;
using __DPL datapar::to_canonical;
using __DPL datapar::to_signed;
using __DPL datapar::to_tuple_like;
using __DPL datapar::to_unsigned;
// NOLINTEND(misc-unused-using-decls)
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
