// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/fwd.h"

#include "dpl/expressions.h" // IWYU pragma: export

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_EXPORT_BEGIN

class __UTL_ABI_PUBLIC binary_vector_expression :
    public expression_interface<vector>,
    public binary_expression_tag {
protected:
    __DPL_HIDE_FROM_ABI ~binary_vector_expression() = default;
};

class __UTL_ABI_PUBLIC unary_vector_expression :
    public expression_interface<vector>,
    public unary_expression_tag {
protected:
    __DPL_HIDE_FROM_ABI ~unary_vector_expression() = default;
};

class __UTL_ABI_PUBLIC binary_scalar_expression :
    public expression_interface<scalar>,
    public binary_expression_tag {
protected:
    __DPL_HIDE_FROM_ABI ~binary_scalar_expression() = default;
};

class __UTL_ABI_PUBLIC unary_scalar_expression :
    public expression_interface<scalar>,
    public unary_expression_tag {
protected:
    __DPL_HIDE_FROM_ABI ~unary_scalar_expression() = default;
};

class __UTL_ABI_PUBLIC binary_vector_mask_expression :
    public expression_interface<vector_mask>,
    public binary_expression_tag {
protected:
    __DPL_HIDE_FROM_ABI ~binary_vector_expression() = default;
};

class __UTL_ABI_PUBLIC unary_vector_mask_expression :
    public expression_interface<vector_mask>,
    public unary_expression_tag {
protected:
    __DPL_HIDE_FROM_ABI ~unary_vector_expression() = default;
};

class __UTL_ABI_PUBLIC binary_scalar_mask_expression :
    public expression_interface<scalar_mask>,
    public binary_expression_tag {
protected:
    __DPL_HIDE_FROM_ABI ~binary_scalar_expression() = default;
};

class __UTL_ABI_PUBLIC unary_scalar_mask_expression :
    public expression_interface<scalar_mask>,
    public unary_expression_tag {
protected:
    __DPL_HIDE_FROM_ABI ~unary_scalar_expression() = default;
};

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_DEFAULT_NAMESPACE_END
