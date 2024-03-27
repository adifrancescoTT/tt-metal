// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_expm1.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_expm1_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::expm1_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_expm1(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>
                                (ckernel::sfpu::calculate_expm1<APPROXIMATE>,
                                ckernel::sfpu::calculate_expm1<APPROXIMATE>,
                                dst_index, vector_mode);
}

}
