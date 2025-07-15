// coding=utf-8
//
// SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "disco.h"

namespace disco_kernels {

    // forward kernel meta
    torch::Tensor disco_meta_fwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
        torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo);

    // backward kernel meta
    torch::Tensor disco_meta_bwd(torch::Tensor inp, torch::Tensor roff_idx, torch::Tensor ker_idx, torch::Tensor row_idx,
        torch::Tensor col_idx, torch::Tensor val, int64_t K, int64_t Ho, int64_t Wo);

    template <typename scalar_t>
    static void disco_fwd_cpu(int64_t B, int64_t C, int64_t K, int64_t Hi, int64_t Wi, int64_t Ho, int64_t Wo, int64_t nnz,
        const torch::PackedTensorAccessor32<scalar_t, 4> inp,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> ker_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> row_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx,
        const torch::PackedTensorAccessor32<scalar_t, 1> vals,
        torch::PackedTensorAccessor32<scalar_t, 5> out) {

        int64_t pscale = static_cast<int64_t>(Wi / Wo);

        // loop over matrix entries
        for (int64_t b = 0; b < B; b++) {
            for (int64_t c = 0; c < C; c++) {

                for (int64_t r = 0; r < nnz; r++) {

                    // COO format, we can optimize later
                    int64_t ho = row_idx[r];
                    int64_t ker = ker_idx[r];
                    int64_t col = col_idx[r];
                    scalar_t val = vals[r];

                    int64_t wi = static_cast<int64_t>(col % Wi);
                    int64_t hi = static_cast<int64_t>(col / Wi);

                    for (int64_t wo = 0; wo < Wo; wo++) {
                        // compute shifted w
                        int64_t wipp = static_cast<int64_t>((wi + pscale * wo) % Wi);
                        out[b][c][ker][ho][wo] += val * inp[b][c][hi][wipp];
                    }
                }
            }
        }
    }

    template <typename scalar_t>
    static void disco_bwd_cpu(int64_t B, int64_t C, int64_t K, int64_t Hi, int64_t Wi, int64_t Ho, int64_t Wo, int64_t nnz,
        const torch::PackedTensorAccessor32<scalar_t, 5> inp,
        const torch::PackedTensorAccessor64<int64_t, 1> roff_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> ker_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> row_idx,
        const torch::PackedTensorAccessor64<int64_t, 1> col_idx,
        const torch::PackedTensorAccessor32<scalar_t, 1> vals,
        torch::PackedTensorAccessor32<scalar_t, 4> out) {

        int64_t pscale = static_cast<int64_t>(Wo / Wi);

        // loop over matrix entries
        for (int64_t b = 0; b < B; b++) {
            for (int64_t c = 0; c < C; c++) {

                for (int64_t r = 0; r < nnz; r++) {

                    // r = ho + Ho * ker
                    int64_t hi = row_idx[r];
                    int64_t ker = ker_idx[r];

                    for (int64_t wi = 0; wi < Wi; wi++) {

                        for (int64_t idc = roff_idx[r]; idc < roff_idx[r+1]; idc++) {
                            int64_t col = col_idx[idc];
                            scalar_t val = vals[idc];

                            // col = wi + Wi * hi -> wi = col % Wi, hi = col // Wi
                            int64_t wo = static_cast<int64_t>(col % Wo);
                            int64_t ho = static_cast<int64_t>(col / Wo);

                            // compute shifted w
                            int64_t wopp = static_cast<int64_t>((wo + pscale * wi) % Wo);
                            out[b][c][ho][wopp] += val * inp[b][c][ker][hi][wi];
                        } 
                    }
                }
            }
        }
    }

}
