# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import torch

from math import ceil

from .shallow_water_equations import ShallowWaterSolver


class PdeDataset(torch.utils.data.Dataset):
    """Custom Dataset class for PDE training data"""

    def __init__(
        self,
        dt,
        nsteps,
        dims=(384, 768),
        grid="equiangular",
        pde="shallow water equations",
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        stream=None,
    ):
        self.num_examples = num_examples
        self.device = device
        self.stream = stream

        self.nlat = dims[0]
        self.nlon = dims[1]

        # number of solver steps used to compute the target
        self.nsteps = nsteps
        self.normalize = normalize

        if pde == "shallow water equations":
            lmax = ceil(self.nlat / 3)
            mmax = lmax
            dt_solver = dt / float(self.nsteps)
            self.solver = ShallowWaterSolver(self.nlat, self.nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid).to(self.device).float()
        else:
            raise NotImplementedError

        self.set_initial_condition(ictype=initial_condition)

        if self.normalize:
            inp0, _ = self._get_sample()
            self.inp_mean = torch.mean(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            self.inp_var = torch.var(inp0, dim=(-1, -2)).reshape(-1, 1, 1)

    def __len__(self):
        length = self.num_examples if self.ictype == "random" else 1
        return length

    def set_initial_condition(self, ictype="random"):
        self.ictype = ictype

    def set_num_examples(self, num_examples=32):
        self.num_examples = num_examples

    def _get_sample(self):
        if self.ictype == "random":
            inp = self.solver.random_initial_condition(mach=0.2)
        elif self.ictype == "galewsky":
            inp = self.solver.galewsky_initial_condition()

        # solve pde for n steps to return the target
        tar = self.solver.timestep(inp, self.nsteps)
        inp = self.solver.spec2grid(inp)
        tar = self.solver.spec2grid(tar)

        return inp, tar

    def __getitem__(self, index):

        # if self.stream is None:
        #     self.stream = torch.cuda.Stream()

        # with torch.cuda.stream(self.stream):
        #     with torch.inference_mode():
        #         with torch.no_grad():
        #             inp, tar = self._get_sample()

        #             if self.normalize:
        #                 inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
        #                 tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        # self.stream.synchronize()

        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
                    tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        return inp.clone(), tar.clone()
