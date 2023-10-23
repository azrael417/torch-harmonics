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

import numpy as np
import torch
import torch.nn as nn
import torch.fft

from .quadrature import *
from .legendre import *

def _sht_r2c(x: torch.Tensor, pct: torch.Tensor) -> torch.Tensor:
    
    # apply real fft in the longitudinal direction
    x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

    # do the Legendre-Gauss quadrature
    x = torch.view_as_real(x)

    # get shapes
    mmax = pct.shape[0]
    lmax = pct.shape[1]
    out_shape = list(x.size())
    out_shape[-3] = lmax
    out_shape[-2] = mmax
    xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

    # contraction
    xout[...] = torch.einsum('...kmr,mlk->...lmr', x[..., :mmax, :], pct.to(x.dtype))
    x = torch.view_as_complex(xout)
    
    return x


def _isht_c2r(x: torch.Tensor, pct: torch.Tensor, nlon: int) -> torch.Tensor:

    # Evaluate associated Legendre functions on the output nodes
    x = torch.view_as_real(x)
    
    xs = torch.einsum('...lms, mlk->...kms', x, pct.to(x.dtype) ).contiguous()
    
    # apply the inverse (real) FFT                                                                                                                                                      
    x = torch.view_as_complex(xs)
    x = torch.fft.irfft(x, n=nlon, dim=-1, norm="forward")

    return x

    
class _RealFwdSHT(torch.autograd.Function):
    """Pass the input to the parallel region."""
    @staticmethod
    def symbolic(graph, input_, comm_id_):
        return input_

    @staticmethod
    def forward(ctx, x: torch.Tensor, pctw: torch.Tensor) -> torch.Tensor:
        # save some things for BWD
        ctx.nlon = x.shape[-1]
        ctx.save_for_backward(pctw)

        # do fwd transform
        y = _sht_r2c(x, pctw)
        
        return y

    @staticmethod
    def backward(ctx, go: torch.Tensor) -> torch.Tensor:
        pctw, = ctx.saved_tensors
        nlon = ctx.nlon

        # conjugate input
        go = go.conj()

        # do bwd transform
        gi = _isht_c2r(go, pctw, nlon)
        
        return gi, None, None

    
class _RealBwdSHT(torch.autograd.Function):
    """Pass the input to the parallel region."""
    @staticmethod
    def symbolic(graph, input_, comm_id_):
        return input_

    @staticmethod
    def forward(ctx, x: torch.Tensor, pct: torch.Tensor, nlon: int) -> torch.Tensor:

        # save some things for BWD
        ctx.save_for_backward(pct)

        # do bwd transform
        y = _isht_c2r(x, pct, nlon)

        return y

    @staticmethod
    def backward(ctx, go: torch.Tensor) -> torch.Tensor:
        pct, = ctx.saved_tensors

        # do bwd transform
        gi = _sht_r2c(go, pct)

        # conjugate
        gi = gi.conj()

        return gi, None, None, None
    

class RealSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        r"""
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase


        # TODO: include assertions regarding the dimensions
        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        # combine quadrature weights with the legendre weights
        weights = torch.from_numpy(w)
        pct = precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase, inverse=False)
        pctw =  torch.einsum('mlk,k->mlk', pct, weights).contiguous()

        # make sure we do not compute the gradients with respect to the Ylm weights
        pctw.requires_grad_ = False

        # remember quadrature weights
        self.register_buffer('pctw', pctw, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert(x.shape[-2] == self.nlat)
        assert(x.shape[-1] == self.nlon)

        return _RealFwdSHT.apply(x, self.pctw)


class InverseRealSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

	# determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # combine quadrature weights with the legendre weights
        pct = precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase, inverse=True)
                                 
        # make sure we do not compute the gradients with respect to the Ylm weights
        pct.requires_grad_ = False

        # remember quadrature weights
        self.register_buffer('pct', pct, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[-2] == self.lmax)
        assert(x.shape[-1] == self.mmax)

        return _RealBwdSHT.apply(x, self.pct, self.nlon)


class RealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        r"""
        Initializes the vector SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: type of grid the data lives on
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        weights = torch.from_numpy(w)
        dpct = precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        
        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)
        norm_factor = 1. / l / (l+1)
        norm_factor[0] = 1.
        weights = torch.einsum('dmlk,k,l->dmlk', dpct, weights, norm_factor)
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(len(x.shape) >= 3)

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
        
        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)
        
        # distributed contraction: fork
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction - spheroidal component
        # real component
        xout[..., 0, :, :, 0] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :self.mmax, 0], self.weights[0].to(x.dtype)) \
                                - torch.einsum('...km,mlk->...lm', x[..., 1, :, :self.mmax, 1], self.weights[1].to(x.dtype)) 

        # iamg component
        xout[..., 0, :, :, 1] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :self.mmax, 1], self.weights[0].to(x.dtype)) \
                                + torch.einsum('...km,mlk->...lm', x[..., 1, :, :self.mmax, 0], self.weights[1].to(x.dtype)) 

        # contraction - toroidal component
        # real component
        xout[..., 1, :, :, 0] = - torch.einsum('...km,mlk->...lm', x[..., 0, :, :self.mmax, 1], self.weights[1].to(x.dtype)) \
                                - torch.einsum('...km,mlk->...lm', x[..., 1, :, :self.mmax, 0], self.weights[0].to(x.dtype)) 
        # imag component
        xout[..., 1, :, :, 1] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :self.mmax, 0], self.weights[1].to(x.dtype)) \
                                - torch.einsum('...km,mlk->...lm', x[..., 1, :, :self.mmax, 1], self.weights[0].to(x.dtype)) 

        return torch.view_as_complex(xout)


class InverseRealVectorSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    
    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """
    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        dpct = precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # register weights
        self.register_buffer('dpct', dpct, persistent=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[-2] == self.lmax)
        assert(x.shape[-1] == self.mmax)
        
        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # contraction - spheroidal component
        # real component
        srl =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[0].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[1].to(x.dtype)) 
        # iamg component
        sim =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[0].to(x.dtype)) \
              + torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[1].to(x.dtype)) 

        # contraction - toroidal component
        # real component
        trl = - torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[1].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[0].to(x.dtype)) 
        # imag component
        tim =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[1].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[0].to(x.dtype)) 
        
        # reassemble
        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)

        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x
