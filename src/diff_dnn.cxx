/*
 * MicroHH
 * Copyright (c) 2011-2020 Chiel van Heerwaarden
 * Copyright (c) 2011-2020 Thijs Heus
 * Copyright (c) 2014-2020 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

#include "grid.h"
#include "fields.h"
#include "master.h"
#include "defines.h"
#include "constants.h"
#include "monin_obukhov.h"
#include "thermo.h"
#include "boundary.h"
#include "stats.h"
#include "fast_math.h"

#include "diff_dnn.h"

namespace
{
    namespace most = Monin_obukhov;
    namespace fm = Fast_math;

    enum class Surface_model {Enabled, Disabled};

    template<typename TF>
    void molecular_diff_c(TF* restrict at, const TF* restrict a, const TF visc,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk, const TF dx, const TF dy, const TF* restrict dzi, const TF* restrict dzhi)
    {
        const int ii = 1;
        const double dxidxi = 1/(dx*dx);
        const double dyidyi = 1/(dy*dy);

        for (int k=kstart; k<kend; k++)
            for (int j=jstart; j<jend; j++)
                #pragma ivdep
                for (int i=istart; i<iend; i++)
                {
                    const int ijk = i + j*jj + k*kk;
                    at[ijk] += visc * (
                            + ( (a[ijk+ii] - a[ijk   ])
                              - (a[ijk   ] - a[ijk-ii]) ) * dxidxi
                            + ( (a[ijk+jj] - a[ijk   ])
                              - (a[ijk   ] - a[ijk-jj]) ) * dyidyi
                            + ( (a[ijk+kk] - a[ijk   ]) * dzhi[k+1]
                              - (a[ijk   ] - a[ijk-kk]) * dzhi[k]   ) * dzi[k] );
                }
    }

    template<typename TF>
    void molecular_diff_w(TF* restrict wt, const TF* restrict w, const TF visc,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk, const TF dx, const TF dy, const TF* restrict dzi, const TF* restrict dzhi)
    {
        const int ii = 1;
        const double dxidxi = 1/(dx*dx);
        const double dyidyi = 1/(dy*dy);

        for (int k=kstart+1; k<kend; k++)
            for (int j=jstart; j<jend; j++)
                #pragma ivdep
                for (int i=istart; i<iend; i++)
                {
                    const int ijk = i + j*jj + k*kk;
                    wt[ijk] += visc * (
                            + ( (w[ijk+ii] - w[ijk   ])
                              - (w[ijk   ] - w[ijk-ii]) ) * dxidxi
                            + ( (w[ijk+jj] - w[ijk   ])
                              - (w[ijk   ] - w[ijk-jj]) ) * dyidyi
                            + ( (w[ijk+kk] - w[ijk   ]) * dzi[k]
                              - (w[ijk   ] - w[ijk-kk]) * dzi[k-1] ) * dzhi[k] );
                }
    }
    
    template <typename TF, Surface_model surface_model>
    void calc_strain2(
            TF* const restrict strain2,
            const TF* const restrict u,
            const TF* const restrict v,
            const TF* const restrict w,
            const TF* const restrict ugradbot,
            const TF* const restrict vgradbot,
            const TF* const restrict z,
            const TF* const restrict dzi,
            const TF* const restrict dzhi,
            const TF dxi, const TF dyi,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        const int ii = 1;
        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        const TF zsl = z[kstart];

        // If the wall isn't resolved, calculate du/dz and dv/dz at lowest grid height using MO
        if (surface_model == Surface_model::Enabled)
        {
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;

                    strain2[ijk] = TF(2.)*(
                            // du/dx + du/dx
                            + fm::pow2((u[ijk+ii]-u[ijk])*dxi)

                            // dv/dy + dv/dy
                            + fm::pow2((v[ijk+jj]-v[ijk])*dyi)

                            // dw/dz + dw/dz
                            + fm::pow2((w[ijk+kk]-w[ijk])*dzi[kstart])

                            // du/dy + dv/dx
                            + TF(0.125)*fm::pow2((u[ijk      ]-u[ijk   -jj])*dyi  + (v[ijk      ]-v[ijk-ii   ])*dxi)
                            + TF(0.125)*fm::pow2((u[ijk+ii   ]-u[ijk+ii-jj])*dyi  + (v[ijk+ii   ]-v[ijk      ])*dxi)
                            + TF(0.125)*fm::pow2((u[ijk   +jj]-u[ijk      ])*dyi  + (v[ijk   +jj]-v[ijk-ii+jj])*dxi)
                            + TF(0.125)*fm::pow2((u[ijk+ii+jj]-u[ijk+ii   ])*dyi  + (v[ijk+ii+jj]-v[ijk   +jj])*dxi)

                            // du/dz
                            + TF(0.5) * fm::pow2(ugradbot[ij])

                            // dw/dx
                            + TF(0.125)*fm::pow2((w[ijk      ]-w[ijk-ii   ])*dxi)
                            + TF(0.125)*fm::pow2((w[ijk+ii   ]-w[ijk      ])*dxi)
                            + TF(0.125)*fm::pow2((w[ijk   +kk]-w[ijk-ii+kk])*dxi)
                            + TF(0.125)*fm::pow2((w[ijk+ii+kk]-w[ijk   +kk])*dxi)

                            // dv/dz
                            + TF(0.5) * fm::pow2(vgradbot[ij])

                            // dw/dy
                            + TF(0.125)*fm::pow2((w[ijk      ]-w[ijk-jj   ])*dyi)
                            + TF(0.125)*fm::pow2((w[ijk+jj   ]-w[ijk      ])*dyi)
                            + TF(0.125)*fm::pow2((w[ijk   +kk]-w[ijk-jj+kk])*dyi)
                            + TF(0.125)*fm::pow2((w[ijk+jj+kk]-w[ijk   +kk])*dyi) );

                    // add a small number to avoid zero divisions
                    strain2[ijk] += Constants::dsmall;
                }
        }

        for (int k=kstart+k_offset; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    strain2[ijk] = TF(2.)*(
                                   // du/dx + du/dx
                                   + fm::pow2((u[ijk+ii]-u[ijk])*dxi)

                                   // dv/dy + dv/dy
                                   + fm::pow2((v[ijk+jj]-v[ijk])*dyi)

                                   // dw/dz + dw/dz
                                   + fm::pow2((w[ijk+kk]-w[ijk])*dzi[k])

                                   // du/dy + dv/dx
                                   + TF(0.125)*fm::pow2((u[ijk      ]-u[ijk   -jj])*dyi  + (v[ijk      ]-v[ijk-ii   ])*dxi)
                                   + TF(0.125)*fm::pow2((u[ijk+ii   ]-u[ijk+ii-jj])*dyi  + (v[ijk+ii   ]-v[ijk      ])*dxi)
                                   + TF(0.125)*fm::pow2((u[ijk   +jj]-u[ijk      ])*dyi  + (v[ijk   +jj]-v[ijk-ii+jj])*dxi)
                                   + TF(0.125)*fm::pow2((u[ijk+ii+jj]-u[ijk+ii   ])*dyi  + (v[ijk+ii+jj]-v[ijk   +jj])*dxi)

                                   // du/dz + dw/dx
                                   + TF(0.125)*fm::pow2((u[ijk      ]-u[ijk   -kk])*dzhi[k  ] + (w[ijk      ]-w[ijk-ii   ])*dxi)
                                   + TF(0.125)*fm::pow2((u[ijk+ii   ]-u[ijk+ii-kk])*dzhi[k  ] + (w[ijk+ii   ]-w[ijk      ])*dxi)
                                   + TF(0.125)*fm::pow2((u[ijk   +kk]-u[ijk      ])*dzhi[k+1] + (w[ijk   +kk]-w[ijk-ii+kk])*dxi)
                                   + TF(0.125)*fm::pow2((u[ijk+ii+kk]-u[ijk+ii   ])*dzhi[k+1] + (w[ijk+ii+kk]-w[ijk   +kk])*dxi)

                                   // dv/dz + dw/dy
                                   + TF(0.125)*fm::pow2((v[ijk      ]-v[ijk   -kk])*dzhi[k  ] + (w[ijk      ]-w[ijk-jj   ])*dyi)
                                   + TF(0.125)*fm::pow2((v[ijk+jj   ]-v[ijk+jj-kk])*dzhi[k  ] + (w[ijk+jj   ]-w[ijk      ])*dyi)
                                   + TF(0.125)*fm::pow2((v[ijk   +kk]-v[ijk      ])*dzhi[k+1] + (w[ijk   +kk]-w[ijk-jj+kk])*dyi)
                                   + TF(0.125)*fm::pow2((v[ijk+jj+kk]-v[ijk+jj   ])*dzhi[k+1] + (w[ijk+jj+kk]-w[ijk   +kk])*dyi) );

                    // Add a small number to avoid zero divisions.
                    strain2[ijk] += Constants::dsmall;
                }
    }

    template <typename TF, Surface_model surface_model>
    void calc_evisc_neutral(
            TF* const restrict evisc,
            const TF* const restrict u,
            const TF* const restrict v,
            const TF* const restrict w,
            const TF* const restrict ufluxbot,
            const TF* const restrict vfluxbot,
            const TF* const restrict z,
            const TF* const restrict dz,
            const TF* const restrict dzhi,
            const TF* const restrict z0m,
            const TF dx, const TF dy, const TF zsize,
            const TF cs, const TF visc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells, const int ijcells,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int jj = icells;
        const int kk = ijcells;

        // Wall damping constant.
        constexpr TF n_mason = TF(1.);
        constexpr TF A_vandriest = TF(26.);

        if (surface_model == Surface_model::Disabled)
        {
            for (int k=kstart; k<kend; ++k)
            {
                // const TF mlen_wall = Constants::kappa<TF>*std::min(z[k], zsize-z[k]);
                const TF mlen_smag = cs*std::pow(dx*dy*dz[k], TF(1./3.));

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk_bot = i + j*jj + kstart*kk;
                        const int ijk_top = i + j*jj + kend*kk;

                        const TF u_tau_bot = std::pow(
                                fm::pow2( visc*(u[ijk_bot] - u[ijk_bot-kk] )*dzhi[kstart] )
                              + fm::pow2( visc*(v[ijk_bot] - v[ijk_bot-kk] )*dzhi[kstart] ), TF(0.25) );
                        const TF u_tau_top = std::pow(
                                fm::pow2( visc*(u[ijk_top] - u[ijk_top-kk] )*dzhi[kend] )
                              + fm::pow2( visc*(v[ijk_top] - v[ijk_top-kk] )*dzhi[kend] ), TF(0.25) );

                        const TF fac_bot = TF(1.) - std::exp( -(       z[k] *u_tau_bot) / (A_vandriest*visc) );
                        const TF fac_top = TF(1.) - std::exp( -((zsize-z[k])*u_tau_top) / (A_vandriest*visc) );
                        const TF fac = std::min( fac_bot, fac_top );

                        const int ijk = i + j*jj + k*kk;
                        evisc[ijk] = fm::pow2(fac * mlen_smag) * std::sqrt(evisc[ijk]);
                    }
            }

            // For a resolved wall the viscosity at the wall is needed. For now, assume that the eddy viscosity
            // is mirrored around the surface.
            const int kb = kstart;
            const int kt = kend-1;
            for (int j=0; j<jcells; ++j)
                #pragma ivdep
                for (int i=0; i<icells; ++i)
                {
                    const int ijkb = i + j*jj + kb*kk;
                    const int ijkt = i + j*jj + kt*kk;
                    evisc[ijkb-kk] = evisc[ijkb];
                    evisc[ijkt+kk] = evisc[ijkt];
                }
        }
        else
        {
            for (int k=kstart; k<kend; ++k)
            {
                // Calculate smagorinsky constant times filter width squared, use wall damping according to Mason's paper.
                const TF mlen0 = cs*std::pow(dx*dy*dz[k], TF(1./3.));

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ij  = i + j*jj;
                        const int ijk = i + j*jj + k*kk;

                        // Mason mixing length
                        const TF mlen = std::pow(TF(1.)/(TF(1.)/std::pow(mlen0, n_mason) +
                                    TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m[ij]), n_mason))), TF(1.)/n_mason);

                        evisc[ijk] = fm::pow2(mlen) * std::sqrt(evisc[ijk]);
                    }
            }
        }

        boundary_cyclic.exec(evisc);
    }

    template<typename TF, Surface_model surface_model>
    void calc_evisc(
            TF* const restrict evisc,
            const TF* const restrict u,
            const TF* const restrict v,
            const TF* const restrict w,
            const TF* const restrict N2,
            const TF* const restrict bgradbot,
            const TF* const restrict z,
            const TF* const restrict dz,
            const TF* const restrict dzi,
            const TF* const restrict z0m,
            const TF dx, const TF dy,
            const TF cs, const TF tPr,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells, const int ijcells,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int jj = icells;
        const int kk = ijcells;

        if (surface_model == Surface_model::Disabled)
        {
            for (int k=kstart; k<kend; ++k)
            {
                // calculate smagorinsky constant times filter width squared, do not use wall damping with resolved walls.
                const TF mlen = cs*std::pow(dx*dy*dz[k], TF(1./3.));
                const TF fac = fm::pow2(mlen);

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;

                        // Add the buoyancy production to the TKE
                        TF RitPrratio = N2[ijk] / evisc[ijk] / tPr;
                        RitPrratio = std::min(RitPrratio, TF(1.-Constants::dsmall));

                        evisc[ijk] = fac * std::sqrt(evisc[ijk]) * std::sqrt(TF(1.)-RitPrratio);
                    }
            }

            // For a resolved wall the viscosity at the wall is needed. For now, assume that the eddy viscosity
            // is mirrored over the surface.
            const int kb = kstart;
            const int kt = kend-1;
            for (int j=0; j<jcells; ++j)
                #pragma ivdep
                for (int i=0; i<icells; ++i)
                {
                    const int ijkb = i + j*jj + kb*kk;
                    const int ijkt = i + j*jj + kt*kk;
                    evisc[ijkb-kk] = evisc[ijkb];
                    evisc[ijkt+kk] = evisc[ijkt];
                }
        }
        else
        {
            // Variables for the wall damping.
            const TF n = 2.;

            // Bottom boundary, here strain is fully parametrized using MO.
            // Calculate smagorinsky constant times filter width squared, use wall damping according to Mason.
            const TF mlen0 = cs*std::pow(dx*dy*dz[kstart], TF(1./3.));

            for (int j=jstart; j<jend; ++j)
            {
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;

                    // TODO use the thermal expansion coefficient from the input later, what to do if there is no buoyancy?
                    // Add the buoyancy production to the TKE
                    TF RitPrratio = bgradbot[ij] / evisc[ijk] / tPr;
                    RitPrratio = std::min(RitPrratio, TF(1.-Constants::dsmall));

                    // Mason mixing length
                    const TF mlen = std::pow(TF(1.)/(TF(1.)/std::pow(mlen0, n) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[kstart]+z0m[ij]), n))), TF(1.)/n);

                    evisc[ijk] = fm::pow2(mlen) * std::sqrt(evisc[ijk]) * std::sqrt(TF(1.)-RitPrratio);
                }
            }

            for (int k=kstart+1; k<kend; ++k)
            {
                // Calculate smagorinsky constant times filter width squared, use wall damping according to Mason
                const TF mlen0 = cs*std::pow(dx*dy*dz[k], TF(1./3.));

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ij  = i + j*jj;
                        const int ijk = i + j*jj + k*kk;

                        // Add the buoyancy production to the TKE
                        TF RitPrratio = N2[ijk] / evisc[ijk] / tPr;
                        RitPrratio = std::min(RitPrratio, TF(1.-Constants::dsmall));

                        // Mason mixing length
                        const TF mlen = std::pow(TF(1.)/(TF(1.)/std::pow(mlen0, n) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m[ij]), n))), TF(1.)/n);

                        evisc[ijk] = fm::pow2(mlen) * std::sqrt(evisc[ijk]) * std::sqrt(TF(1.)-RitPrratio);
                    }
            }
        }

        boundary_cyclic.exec(evisc);
    }
    
    template <typename TF, Surface_model surface_model>
    void destagger_u(
            TF* const restrict uc,
            const TF* const restrict u,
            const Grid_order spatial_order,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        
        const int ii = 1;
        if (spatial_order == Grid_order::Fourth)
            for (int k=kstart; k<kend; ++k)
                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        uc[ijk] = (u[ijk+ii+ii]+u[ijk-ii])/TF(6.)+(u[ijk+ii]+u[ijk])/TF(3.);
                    }
        else
            for (int k=kstart; k<kend; ++k)
                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        uc[ijk] = TF(0.5)*(u[ijk+ii]+u[ijk]);
                    }
        
        boundary_cyclic.exec(uc);
    }
    
    template <typename TF, Surface_model surface_model>
    void destagger_v(
            TF* const restrict vc,
            const TF* const restrict v,
            const Grid_order spatial_order,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        
        if (spatial_order == Grid_order::Fourth)
            for (int k=kstart; k<kend; ++k)
                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        vc[ijk] = (v[ijk+jj+jj]+v[ijk-jj])/TF(6.)+(v[ijk+jj]+v[ijk])/TF(3.);
                    }
        else
             for (int k=kstart; k<kend; ++k)
                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        vc[ijk] = TF(0.5)*(v[ijk+jj]+v[ijk]);
                    }
        boundary_cyclic.exec(vc);
    }
    
    template <typename TF, Surface_model surface_model>
    void destagger_w(
            TF* const restrict wc,
            const TF* const restrict w,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        
        const int ii = 1;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    wc[ijk] = TF(0.5)*(w[ijk+kk]+w[ijk]);
                }
        boundary_cyclic.exec(wc);
    }
    
    template <typename TF, Surface_model surface_model>
    void calc_TKEh(
            TF* const restrict TKEh,
            const TF* const restrict uc,
            const TF* const restrict vc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int ii = 1;
        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;
        /* Can't use DNN at first grid level, so no need to compute scaling factors here
        if (surface_model == Surface_model::Enabled)
        {
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;
                    
                    TKEh[ijk] = TF(0.25)*(
                                   // dx**2*(du/dx)**2 
                                   + fm::pow2(uc[ijk+ii]-uc[ijk-ii])
                                   // dx**2*(dv/dx)**2
                                   + fm::pow2(vc[ijk+ii]-vc[ijk-ii])
                                   // dy**2*(du/dy)**2
                                   + fm::pow2(uc[ijk+jj]-uc[ijk-jj])
                                   // dy**2*(dv/dy)**2
                                   + fm::pow2(vc[ijk+jj]-vc[ijk-jj])
                                   // dz**2*(du/dz)**2 2nd order differencing, but one-sided at sfc
                                   + fm::pow2(-uc[ijk+2*kk]+TF(4.0)*uc[ijk+kk]-TF(3.0)*uc[ijk])
                                   // dz**2*(dv/dz)**2 " "
                                   + fm::pow2(-vc[ijk+2*kk]+TF(4.0)*vc[ijk+kk]-TF(3.0)*vc[ijk]));
                    //TKEh[ijk] += Constants::dsmall;
                }
        }*/

        for (int k=kstart+k_offset; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    TKEh[ijk] = TF(0.25)*(
                                   // dx**2*(du/dx)**2 
                                   + fm::pow2(uc[ijk+ii]-uc[ijk-ii])
                                   // dx**2*(dv/dx)**2
                                   + fm::pow2(vc[ijk+ii]-vc[ijk-ii])
                                   // dy**2*(du/dy)**2
                                   + fm::pow2(uc[ijk+jj]-uc[ijk-jj])
                                   // dy**2*(dv/dy)**2
                                   + fm::pow2(vc[ijk+jj]-vc[ijk-jj])
                                   // dz**2*(du/dz)**2 
                                   + fm::pow2(uc[ijk+kk]-uc[ijk-kk])
                                   // dz**2*(dv/dz)**2
                                   + fm::pow2(vc[ijk+kk]-vc[ijk-kk]) );

                    // Add a small number to avoid zero divisions.
                    //TKEh[ijk] += Constants::dsmall;
                }
        boundary_cyclic.exec(TKEh);
    }
    
    template <typename TF, Surface_model surface_model>
    void calc_TKEv(
            TF* const restrict TKEv,
            const TF* const restrict wc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int ii = 1;
        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;
        /* Can't use DNN at first grid level, so no need to compute scaling factors here
        if (surface_model == Surface_model::Enabled)
        {
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;
                    
                    TKEv[ijk] = TF(0.25)*(
                                   // dx**2*(dw/dx)**2 
                                   + fm::pow2(wc[ijk+ii]-wc[ijk-ii])
                                   // dy**2*(dw/dy)**2
                                   + fm::pow2(wc[ijk+jj]-wc[ijk-jj])
                                   // dz**2*(dw/dz)**2 2nd order differencing, but one-sided at sfc
                                   + fm::pow2(-wc[ijk+2*kk]+TF(4.0)*wc[ijk+kk]-TF(3.0)*wc[ijk]));
                    //TKEv[ijk] += Constants::dsmall;
                }
        }*/
                
        for (int k=kstart+k_offset; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    TKEv[ijk] = TF(0.25)*(
                                   // dx**2*(db/dx)**2 
                                   + fm::pow2(wc[ijk+ii]-wc[ijk-ii])
                                   // dy**2*(db/dy)**2
                                   + fm::pow2(wc[ijk+jj]-wc[ijk-jj])
                                   // dz**2*(db/dz)**2
                                   + fm::pow2(wc[ijk+kk]-wc[ijk-kk]));

                    // Add a small number to avoid zero divisions.
                    //TKEv[ijk] += Constants::dsmall;
                }
        boundary_cyclic.exec(TKEv);
    }
        
    template <typename TF, Surface_model surface_model>
    void calc_TPE(
            TF* const restrict TPE,
            const TF* const restrict b,
            const TF* const restrict N2,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int ii = 1;
        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;
        /* Can't use DNN at first grid level, so no need to compute scaling factors here
        if (surface_model == Surface_model::Enabled)
        {
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;
                    
                    TPE[ijk] = TF(0.25)*(
                                   // dx**2*(db/dx)**2 
                                   + fm::pow2(b[ijk+ii]-b[ijk-ii])
                                   // dy**2*(db/dy)**2
                                   + fm::pow2(b[ijk+jj]-b[ijk-jj])
                                   // dz**2*(db/dz)**2 2nd order differencing, but one-sided at sfc
                                   + fm::pow2(-b[ijk+2*kk]+TF(4.0)*b[ijk+kk]-TF(3.0)*b[ijk]))/N2[ijk];
                    //TPE[ijk] += Constants::dsmall;
                }
        }*/

        for (int k=kstart+k_offset; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    TPE[ijk] = TF(0.25)*(
                                   // dx**2*(db/dx)**2 
                                   + fm::pow2(b[ijk+ii]-b[ijk-ii])
                                   // dy**2*(db/dy)**2
                                   + fm::pow2(b[ijk+jj]-b[ijk-jj])
                                   // dz**2*(db/dz)**2
                                   + fm::pow2(b[ijk+kk]-b[ijk-kk]))/N2[ijk];

                    // Add a small number to avoid zero divisions.
                    //TPE[ijk] += Constants::dsmall;
                }
        boundary_cyclic.exec(TPE);
    }
    
    template <typename TF, Surface_model surface_model>
    at::Tensor calc_Tau(
            torch::jit::script::Module dnn,
            const TF* const restrict uc,
            const TF* const restrict vc,    
            const TF* const restrict wc,
            const TF* const restrict b,
            const TF* const restrict TKEh,
            const TF* const restrict TKEv,
            const TF* const restrict TPE,
            const TF* const restrict dz,
            const int nh,
            const int ncells,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        const int ii = 1;
        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;
        
        const int nv = 3; // Vertical levels of input to DNN is fixed
        const int iv = nv/2; 
        const int ih = nh/2;
        const int nbox = nv*nh*nh;
        const int kendBL = kstart+(3*(kend-kstart))/4;
        
        const int jjbatch = iend-istart;
        const int kkbatch = jjbatch*(jend-jstart);
        
        const int nbatch = kkbatch*(kendBL-1-kstart-k_offset);
        
        at::Tensor x = torch::zeros({1, 4*nv, nh, nh}); // Number of input variables, 4, is fixed: u,v,w,b       
        at::Tensor Tau = torch::zeros({nbatch, 6});
        
        for (int k=kstart+k_offset; k<kendBL-1; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    //const int ijkbatch = 1+i-istart + (j-jstart)*jjbatch + (k-kstart-k_offset)*kkbatch;
                                       
                    TF ubar = 0;
                    TF vbar = 0;
                    TF wbar = 0;
                    TF bbar = 0;
                    for (int ix=-ih; ix<=ih; ix++)
                        for (int iy=-ih; iy<=ih; iy++)
                            for (int iz=-iv; iz<=iv; iz++)
                            {
                                ubar += uc[ijk+ix*ii+iy*jj+iz*kk];
                                vbar += vc[ijk+ix*ii+iy*jj+iz*kk];
                                wbar += wc[ijk+ix*ii+iy*jj+iz*kk];
                                bbar +=  b[ijk+ix*ii+iy*jj+iz*kk];
                             }
                    ubar=ubar/nbox;
                    vbar=vbar/nbox;
                    wbar=wbar/nbox;
                    bbar=bbar/nbox;

                    TF rootki = std::pow(TKEh[ijk]+TKEv[ijk]+Constants::dtiny,-0.5);
                    TF rootkvi = std::pow(TKEv[ijk]+Constants::dtiny,-0.5);
                    TF bScalei = dz[k]/(TPE[ijk]+Constants::dtiny);
                    for (int iz=-iv; iz<=iv; iz++)
                        for (int ix=-ih; ix<=ih; ++ix)
                                for (int iy=-ih; iy<=ih; ++iy)
                                {                                    
                                    x.index_put_({0, 2*(iz+iv),ih+ix,ih+iy}, (uc[ijk+ix*ii+iy*jj+iz*kk]-ubar)*rootki);
                                    x.index_put_({0, 2*(iz+iv)+1,ih+ix,ih+iy}, (vc[ijk+ix*ii+iy*jj+iz*kk]-vbar)*rootki);
                                    x.index_put_({0, 2*nv+(iz+iv),ih+ix,ih+iy}, (wc[ijk+ix*ii+iy*jj+iz*kk]-wbar)*rootkvi);
                                    x.index_put_({0, 3*nv+(iz+iv),ih+ix,ih+iy}, (b[ijk+ix*ii+iy*jj+iz*kk]-bbar)*bScalei);
                                }
                    /*if (ijkbatch==4){std::cout << x.slice(0, 0,4) << std::endl;}
                    if (ijk==(iend/2+ (jend/2)*jj + (kend/3)*kk)){std::cout << x.slice(0, ijk,ijk+1) << std::endl;}*/
                  
                    std::vector<torch::jit::IValue> input;
                    input.push_back(x);
  
                    at::Tensor output = dnn.forward(input).toTensor();
                    const int ijkbatch = i-istart + (j-jstart)*jjbatch + (k-kstart-k_offset)*kkbatch;                
                 
                    TF ktot = TKEh[ijk]+TKEv[ijk];
                    TF kv = TKEv[ijk];
                    TF rootkkv =  std::pow(ktot*kv,0.5);
                    
                    /*if (ijkbatch==4) {std::cout << Tau.slice(0, 0, 4) << std::endl;}
                    if (ijk== (iend/2+ (jend/2)*jj + (kend/3)*kk)) {std::cout << Tau.slice(0, ijk,ijk+1) << std::endl;}*/
                    Tau.index_put_({ijkbatch, 0}, output.index({0, 0}) * ktot );
                    Tau.index_put_({ijkbatch, 1}, output.index({0, 1}) * ktot );
                    Tau.index_put_({ijkbatch, 2}, output.index({0, 2}) * rootkkv );
                    Tau.index_put_({ijkbatch, 3}, output.index({0, 3}) * ktot );
                    Tau.index_put_({ijkbatch, 4}, output.index({0, 4}) * rootkkv );
                    Tau.index_put_({ijkbatch, 5}, output.index({0, 5}) * kv );
                    /*if (ijkbatch==04) {std::cout << Tau.slice(0, 0, 4) << std::endl;}
                    if (ijk==(iend/2+ (jend/2)*jj + (kend/3)*kk)) {std::cout << Tau.slice(0, ijk,ijk+1) << std::endl;} */
                    
                }
        //std::cout << Tau << std::endl;        
        
        return Tau.to(torch::kDouble);
    }

    template<typename TF>
    void set_flux(
            TF* const restrict flux_fld,
            TF* const restrict fluxtop,
            const at::Tensor Tau,
            const int dim,
            const TF* const restrict fluxbot,
            const TF* const restrict z,
            const TF* const restrict zh,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int ijcells,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        //using namespace torch::indexing;       
        //const TF* flux = Tau.slice(1, dim, dim+1).contiguous().data_ptr<TF>();
        auto tau = Tau.accessor<TF,2>();

        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;
        const int jjbatch = iend-istart;
        const int kkbatch = jjbatch*(jend-jstart);
        const int kendBL = kstart+(3*(kend-kstart))/4;
        
        for (int k=kstart+1; k<kendBL-1; ++k)    
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const int ijkbatch = i-istart + (j-jstart)*jjbatch + (k-kstart-1)*kkbatch;

                    flux_fld[ijk] = tau[ijkbatch][dim];//if(dim==0 or dim==3 or dim==5){flux_fld[ijk] = std::max(0,flux_fld[ijk]);
                }
       
        // First half-level 
        if (dim==2){
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {    
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kstart)*kk;
                    // Linearly interpolating Tau_13 from surface flux values
                    flux_fld[ijk] = TF(0.5)*(fluxbot[ij]+fluxbot[ij+ii]) // destaggering to cell centers
                                    +(z[kstart]-zh[kstart])*(flux_fld[ijk+kk]-TF(0.5)*(fluxbot[ij]+fluxbot[ij+ii]))
                                        /(z[kstart+1]-zh[kstart]);
                }}
        
        else if (dim==4){
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {    
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kstart)*kk;
                    // Linearly interpolating Tau_23 from surface flux values
                    flux_fld[ijk] = TF(0.5)*(fluxbot[ij]+fluxbot[ij+jj]) // destaggering to cell centers
                                    +(z[kstart]-zh[kstart])*(flux_fld[ijk+kk]-TF(0.5)*(fluxbot[ij]+fluxbot[ij+jj]))
                                        /(z[kstart+1]-zh[kstart]);
                }}
        else{
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {    
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kstart)*kk;
                    // Linearly interpolating all other components from tau11=tau12=tau22=tau33=0 at surface 
                    flux_fld[ijk] = (z[kstart]-zh[kstart])*(flux_fld[ijk+kk])/(z[kstart+1]-zh[kstart]);
                }}
        
        // Top BC
        for (int j=jstart; j<jend; ++j)
                #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + (kendBL-1)*kk;
                //const int ijk = i + j*jj + kendBL*kk;
                
                // Can't compute at top, so set gradient to zero
                flux_fld[ijk] = flux_fld[ijk-kk]; // these are only levels that get touched in diff_* but could fill all, or
                flux_fld[ijk+kk] = flux_fld[ijk]; // If fluxtop is known somehow, change this to interpolation
                if (dim==2 or dim==4)
                    fluxtop[ij] = flux_fld[ijk]; // 2 is Tau13, 4 is Tau23, this is wrong staggering but never used so fix later
            }
        if(dim==2 or dim==4){boundary_cyclic.exec_2d(fluxtop);}
        boundary_cyclic.exec(flux_fld);
    } 
    
    template <typename TF, Surface_model surface_model>
    void diff_u(
            TF* const restrict ut,
            const TF* const restrict T11,
            const TF* const restrict T12,
            const TF* const restrict T13,
            const TF* const restrict z,
            const TF* const restrict zh,
            const TF dxi, const TF dyi,
            const TF* const restrict fluxbot,
            const TF* const restrict fluxtop,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        constexpr int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;
        const int kendBL = kstart+(3*(kend-kstart))/4;
        const int ii = 1;

        if (surface_model == Surface_model::Enabled)
        {
            // bottom boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    // first half level
                    const int ijk = i + j*jj + kstart*kk;
                    
                    ut[ijk] +=
                            // -dTau11/dx = 0 in surface layer by horizontal homogeneity assumption
                            -dxi*(T11[ijk+ii]-T11[ijk])
                            // -dTau12/dy = 0 in surface layer by horizontal homogeneity assumption
                            -TF(0.25)*dyi*(T12[ijk+jj]+T12[ijk+ii+jj]-T12[ijk-jj]-T12[ijk+ii-jj]) 
                             // -dTau13/dz
                            -(TF(0.5)*(T13[ijk+kk]+T13[ijk-ii+kk])-fluxbot[ij])/(z[kstart+1]-zh[kstart]);
                    
                    // second half level, interpolation happens in set_flux so don't handle separately here        
                }
        }

        for (int k=kstart+k_offset; k<kendBL; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    ut[ijk] +=
                             // -dTau11/dx
                            -dxi*(T11[ijk]-T11[ijk-ii])
                            // -dTau12/dy
                            -TF(0.25)*dyi*(T12[ijk+jj]+T12[ijk-ii+jj]-T12[ijk-jj]-T12[ijk-ii-jj]) 
                             // -dTau13/dz
                            -TF(0.5)*(T13[ijk+kk]+T13[ijk-ii+kk]-T13[ijk-kk]-T13[ijk-ii-kk])/(z[k+1]-z[k-1]); 
                }
        
         // DNN turned off above boundary layer
             /*for (int k=kendBL; k<kend-1; ++k)
               ...
             const int ijk = i + j*jj + (kend-1)*kk;
                   ut[ijk] += ... -(fluxtop[ij]-TF(0.5)*(T13[ijk]+T13[ijk+ii]))/(zh[kend]-z[kend-1]);} // -dTau13/dz
            */              
    }

    template <typename TF, Surface_model surface_model>
    void diff_v(
            TF* const restrict vt,
            const TF* const restrict T12,
            const TF* const restrict T22,
            const TF* const restrict T23,
            const TF* const restrict z,
            const TF* const restrict zh,
            const TF dxi, const TF dyi,
            const TF* const restrict fluxbot,
            const TF* const restrict fluxtop,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)

    {
        constexpr int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;
        const int kendBL = kstart+(3*(kend-kstart))/4;
        
        const int ii = 1;

        if (surface_model == Surface_model::Enabled)
        {
            // bottom boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    int ijk = i + j*jj + kstart*kk;
                    
                    vt[ijk] += 
                            //-dTau21/dx
                            -TF(0.25)*dxi*(T12[ijk+ii]+T12[ijk+ii+jj]-T12[ijk-ii]-T12[ijk-ii+jj])
                            // -dTau22/dy
                            -dyi*(T22[ijk+jj]-T22[ijk]) 
                             // -dTau23/dz
                            -(TF(0.5)*(T23[ijk+kk]+T23[ijk-jj+kk])-fluxbot[ij])/(z[kstart+1]-zh[kstart]);

                    //second half level, interpolation happens in set_flux so don't handle separately here
                }
        }

        for (int k=kstart+k_offset; k<kendBL; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    vt[ijk] +=
                            // -dTau21/dx
                            -TF(0.25)*dxi*(T12[ijk+ii]+T12[ijk+ii-jj]-T12[ijk-ii]-T12[ijk-ii-jj])
                            // -dTau22/dy
                            -dyi*(T22[ijk]-T22[ijk-jj]) 
                             // -dTau23/dz
                            -TF(0.5)*(T23[ijk+kk]+T23[ijk-jj+kk]-T23[ijk-kk]-T23[ijk-jj-kk])/(z[k+1]-z[k-1]); 
                }
         
        // DNN turned off above boundary layer
             /*for (int k=kendBL; k<kend-1; ++k)
               ...
             const int ijk = i + j*jj + (kend-1)*kk;
                    vt[ijk] += -(fluxtop[ij]-TF(0.5)*(T23[ijk]+T23[ijk+jj]))/(zh[kend]-z[kend-1]);} // -dTau23/dz
            */        
    }

    template <typename TF>
    void diff_w(
            TF* const restrict wt,
            const TF* const restrict T13,
            const TF* const restrict T23,
            const TF* const restrict T33,
            const TF* const restrict ufluxbot,
            const TF* const restrict vfluxbot,
            const TF* const restrict z,
            const TF* const restrict zh,
            const TF dxi, const TF dyi,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        const int ii = 1;
        const int kendBL = kstart+(3*(kend-kstart))/4;

        for (int k=kstart+2; k<kendBL; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    wt[ijk] +=
                            // -dTau31/dx
                            -TF(0.25)*dxi*(T13[ijk+ii]+T13[ijk+ii-kk]-T13[ijk-ii]-T13[ijk-ii-kk])
                            // -dTau32/dy
                            -TF(0.25)*dyi*(T23[ijk+jj]+T23[ijk+jj-kk]-T23[ijk-jj]-T23[ijk-jj-kk])
                             // -dTau33/dz
                            -(T33[ijk]-T33[ijk-kk])/(z[k]-z[k-1]); 
                }

        // first above bottom boundary
        // Though we can interpolate using the surface flux values for some of the terms below, 
        // proably best to just assume -dTau31/dx = -dTau32/dy = 0 by horizontal homeogeneity 
        // -dTau33/dz = 0 is harder to justify, but perhaps follows from divergence-free constraints 
        // In any case, just turning off this block of code but saving to turn on later maybe
        if (0)             
        {   for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {    
                    const int ij = i + j*jj;
                    const int ijk = i + j*jj + (kstart+1)*kk;
                    wt[ijk] +=
                            // -dTau31/dx , interpolation happens in set_flux so don't handle separately here
                            -TF(0.25)*dxi*(T13[ijk+ii]+T13[ijk+ii-kk]-T13[ijk-ii]-T13[ijk-ii-kk])
                            // -dTau32/dy , interpolation happens in set_flux so don't handle separately here
                            -TF(0.25)*dyi*(T23[ijk+jj]+T23[ijk+jj-kk]-T23[ijk-jj]-T23[ijk-jj-kk]);
                             // -dTau33/dz  
                            /* Can't do centered differnce, b/c don't have <w'w'> at surface as we do <u'w'>, <v'w'> from MOST
                               One-side difference with zero fluz at surface?
                                   -T33[ijk]/(z[kstart+1]-zh[kstart]);
                                       maybe but probably can't argue <w'w'>=0 at surface by no flux
                                       in the same way we can't argue <u'w'> = 0 at surface by no slip
                                Zero Neumann i.e enforcing dTau33/dz = 0 in surface layer? 
                                    Implement by doing nothing here, 
                                    can't really justify that physically, but also can't justify the linear interpolation 
                            */
                    }
            }
    
        // DNN turned off above boundary layer
             /*for (int k=kendBL; k<kend-1; ++k)   ...
             const int ijk = i + j*jj + (kend-1)*kk; */
    }

    template <typename TF, Surface_model surface_model>
    void diff_c(
            TF* const restrict at,
            const TF* const restrict a,
            const TF* const restrict dzi,
            const TF* const restrict dzhi,
            const TF dxidxi, const TF dyidyi,
            const TF* const restrict evisc,
            const TF* const restrict fluxbot,
            const TF* const restrict fluxtop,
            const TF* const restrict rhoref,
            const TF* const restrict rhorefh,
            const TF tPr, const TF visc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        constexpr int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        const int ii = 1;

        if (surface_model == Surface_model::Enabled)
        {
            // bottom boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;
                    const TF evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii])/tPr + visc;
                    const TF eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ])/tPr + visc;
                    const TF eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj])/tPr + visc;
                    const TF eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ])/tPr + visc;
                    const TF evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk])/tPr + visc;

                    at[ijk] +=
                             + ( evisce*(a[ijk+ii]-a[ijk   ])
                               - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( eviscn*(a[ijk+jj]-a[ijk   ])
                               - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + ( rhorefh[kstart+1] * evisct*(a[ijk+kk]-a[ijk   ])*dzhi[kstart+1]
                               + rhorefh[kstart  ] * fluxbot[ij] ) / rhoref[kstart] * dzi[kstart];
                }

            // top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kend-1)*kk;
                    const TF evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii])/tPr + visc;
                    const TF eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ])/tPr + visc;
                    const TF eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj])/tPr + visc;
                    const TF eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ])/tPr + visc;
                    const TF eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ])/tPr + visc;

                    at[ijk] +=
                             + ( evisce*(a[ijk+ii]-a[ijk   ])
                               - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( eviscn*(a[ijk+jj]-a[ijk   ])
                               - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + (-rhorefh[kend  ] * fluxtop[ij]
                               - rhorefh[kend-1] * eviscb*(a[ijk   ]-a[ijk-kk])*dzhi[kend-1] ) / rhoref[kend-1] * dzi[kend-1];
                }
        }

        for (int k=kstart+k_offset; k<kend-k_offset; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const TF evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii])/tPr + visc;
                    const TF eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ])/tPr + visc;
                    const TF eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj])/tPr + visc;
                    const TF eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ])/tPr + visc;
                    const TF evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk])/tPr + visc;
                    const TF eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ])/tPr + visc;

                    at[ijk] +=
                             + ( evisce*(a[ijk+ii]-a[ijk   ])
                               - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( eviscn*(a[ijk+jj]-a[ijk   ])
                               - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + ( rhorefh[k+1] * evisct*(a[ijk+kk]-a[ijk   ])*dzhi[k+1]
                               - rhorefh[k  ] * eviscb*(a[ijk   ]-a[ijk-kk])*dzhi[k]  ) / rhoref[k] * dzi[k];
                }
    }

    template<typename TF>
    TF calc_dnmul(
            const TF* const restrict evisc,
            const TF* const restrict dzi,
            const TF dxidxi, const TF dyidyi,
            const TF tPr,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        const TF tPrfac_i = TF(1)/std::min(TF(1.), tPr);
        TF dnmul = 0;

        // get the maximum time step for diffusion
        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    dnmul = std::max(dnmul, std::abs(evisc[ijk]*tPrfac_i*(dxidxi + dyidyi + dzi[k]*dzi[k])));
                }

        return dnmul;
    }

    template <typename TF, Surface_model surface_model>
    void calc_diff_flux_c(
            TF* const restrict out,
            const TF* const restrict data,
            const TF* const restrict evisc,
            const TF* const restrict dzhi,
            const TF tPr, const TF visc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        constexpr int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        #pragma omp parallel for
        for (int k=kstart+k_offset; k<(kend+1-k_offset); ++k)
        {
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const TF eviscc = 0.5*(evisc[ijk-kk]+evisc[ijk])/tPr + visc;

                    out[ijk] = - eviscc*(data[ijk] - data[ijk-kk])*dzhi[k];
                }
        }
    }

    template <typename TF, Surface_model surface_model>
    void calc_diff_flux_u(
            TF* const restrict out,
            const TF* const restrict data,
            const TF* const restrict w,
            const TF* const evisc,
            const TF dxi, const TF* const dzhi,
            const TF visc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int ijcells)
    {
        constexpr int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        const int ii = 1;
        #pragma omp parallel for
        for (int k=kstart+k_offset; k<(kend+1-k_offset); ++k)
        {
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*icells + k*ijcells;
                    const TF eviscu = 0.25*(evisc[ijk-ii-ijcells]+evisc[ijk-ii]+evisc[ijk-ijcells]+evisc[ijk]) + visc;
                    out[ijk] = - eviscu*( (data[ijk]-data[ijk-ijcells])*dzhi[k] + (w[ijk]-w[ijk-ii])*dxi );
                }
        }
    }

    template <typename TF, Surface_model surface_model>
    void calc_diff_flux_v(
            TF* const restrict out,
            const TF* const restrict data,
            const TF* const restrict w,
            const TF* const evisc,
            const TF dyi, const TF* const dzhi,
            const TF visc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int ijcells)
    {
        constexpr int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        #pragma omp parallel for
        for (int k=kstart+k_offset; k<(kend+1-k_offset); ++k)
        {
                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*icells + k*ijcells;
                        const TF eviscv = 0.25*(evisc[ijk-icells-ijcells]+evisc[ijk-icells]+evisc[ijk-ijcells]+evisc[ijk]) + visc;
                        out[ijk] = - eviscv*( (data[ijk]-data[ijk-ijcells])*dzhi[k] + (w[ijk]-w[ijk-icells])*dyi );
                    }
        }
    }

    template<typename TF>
    void calc_diff_flux_bc(
            TF* const restrict out,
            const TF* const restrict data,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int k, const int icells, const int ijcells)
    {
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ij  = i + j*icells;
                const int ijk = i + j*icells + k*ijcells;
                out[ijk] = data[ij];
            }
    }

} // End namespace.

template<typename TF>
Diff_dnn<TF>::Diff_dnn(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Boundary<TF>& boundaryin, Input& inputin) :
    Diff<TF>(masterin, gridin, fieldsin, boundaryin, inputin),
    boundary_cyclic(master, grid),
    field3d_operators(master, grid, fields)
{
    auto& gd = grid.get_grid_data();
    dnmax = inputin.get_item<TF>("diff", "dnmax", "", 0.4  );
    cs    = inputin.get_item<TF>("diff", "cs"   , "", 0.23 );
    ce    = inputin.get_item<TF>("diff", "ce"   , "", 0.15 );
    tPr   = inputin.get_item<TF>("diff", "tPr"  , "", 1./3.);
    dnnpath = inputin.get_item<std::string>("diff", "dnnpath"  , "", "C4_midGridReInterp_local_4x1026Re900_4x3078Re2700_1.pt");
    try {dnn = torch::jit::load(dnnpath);} // Deserialize the ScriptModule from a file
    catch (const c10::Error& e) {std::cerr << "error loading the deep neural network\n";}
        
    const std::string group_name = "default";

    fields.init_diagnostic_field("evisc", "Eddy viscosity", "m2 s-1", group_name, gd.sloc);
    fields.init_diagnostic_field("T11", "Turbulent flux of u_1 mom'm in x_1 direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T12", "Turbulent flux of u_1(2) mom'm in x_2(1) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T13", "Turbulent flux of u_1(3) mom'm in x_3(1) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T22", "Turbulent flux of u_2 mom'm in x_2 direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T23", "Turbulent flux of u_2(3) mom'm in x_3(2) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T33", "Turbulent flux of u_3 mom'm in x_3 direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("TKEh", "Proportional to turbulent kinetic energy associated with horizontal velocities as estimated by Taylor expansion", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("TKEv", "Proportional to turbulent kinetic energy associated with vertical velocity as estimated by Taylor expansion", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("TPE", "Proportional to turbulent potential energy as estimated by Taylor expansion of buoyancy", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("uc", "Destaggered u velocity", "m s-1", group_name, gd.sloc);
    fields.init_diagnostic_field("vc", "Destaggered v velocity", "m s-1", group_name, gd.sloc);
    fields.init_diagnostic_field("wc", "Destaggered w velocity", "m s-1", group_name, gd.sloc);

        
/*    if (grid.get_spatial_order() != Grid_order::Second)
        throw std::runtime_error("Diff_dnn only runs with second order grids");*/
}

template<typename TF>
Diff_dnn<TF>::~Diff_dnn()
{
}

template<typename TF>
void Diff_dnn<TF>::init()
{
    boundary_cyclic.init();
}

template<typename TF>
Diffusion_type Diff_dnn<TF>::get_switch() const
{
    return swdiff;
}

#ifndef USECUDA
template<typename TF>
unsigned long Diff_dnn<TF>::get_time_limit(const unsigned long idt, const double dt)
{
    auto& gd = grid.get_grid_data();

    double dnmul = calc_dnmul<TF>(
        fields.sd.at("evisc")->fld.data(),
        gd.dzi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy), tPr,
        gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
        gd.icells, gd.ijcells);
    master.max(&dnmul, 1);

    // Avoid zero division.
    dnmul = std::max(Constants::dsmall, dnmul);

    return idt * dnmax / (dt * dnmul);
}
#endif

#ifndef USECUDA
template<typename TF>
double Diff_dnn<TF>::get_dn(const double dt)
{
    auto& gd = grid.get_grid_data();

    double dnmul = calc_dnmul<TF>(
        fields.sd.at("evisc")->fld.data(),
        gd.dzi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy), tPr,
        gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
        gd.icells, gd.ijcells);
    master.max(&dnmul, 1);

    return dnmul*dt;
}
#endif

template<typename TF>
void Diff_dnn<TF>::create(Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();

    // Get the maximum viscosity
    TF viscmax = fields.visc;
    for (auto& it : fields.sp)
        viscmax = std::max(it.second->visc, viscmax);

    // Calculate time step multiplier for diffusion number
    dnmul = 0;
    for (int k=gd.kstart; k<gd.kend; ++k)
        dnmul = std::max(dnmul, std::abs(viscmax * (1./(gd.dx*gd.dx) + 1./(gd.dy*gd.dy) + 1./(gd.dz[k]*gd.dz[k]))));

    create_stats(stats);
}

#ifndef USECUDA
template<typename TF>
void Diff_dnn<TF>::exec(Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();
    
    if (boundary.get_switch() != "default")
    {    
    set_flux<TF>(fields.sd.at("T11")->fld.data(),nullptr,
                    Tau,0,
                    nullptr,
                    gd.z.data(), gd.zh.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    set_flux<TF>(fields.sd.at("T12")->fld.data(),nullptr,
                    Tau,1,
                    nullptr,
                    gd.z.data(), gd.zh.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    set_flux<TF>(fields.sd.at("T13")->fld.data(),fields.mp.at("u")->flux_top.data(),
                    Tau,2,
                    fields.mp.at("u")->flux_bot.data(),
                    gd.z.data(), gd.zh.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    set_flux<TF>(fields.sd.at("T22")->fld.data(),nullptr,
                    Tau,3,
                    nullptr,
                    gd.z.data(), gd.zh.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    set_flux<TF>(fields.sd.at("T23")->fld.data(),fields.mp.at("v")->flux_top.data(),
                    Tau,4,
                    fields.mp.at("v")->flux_bot.data(),
                    gd.z.data(), gd.zh.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    set_flux<TF>(fields.sd.at("T33")->fld.data(),nullptr,
                    Tau,5,
                    nullptr,
                    gd.z.data(), gd.zh.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic); 

    diff_u<TF, Surface_model::Enabled>(
                fields.mt.at("u")->fld.data(),
                fields.sd.at("T11")->fld.data(), fields.sd.at("T12")->fld.data(), fields.sd.at("T13")->fld.data(),
                gd.z.data(), gd.zh.data(), 1./gd.dx, 1./gd.dy,
                //fields.sd.at("evisc")->fld.data(),
                fields.mp.at("u")->flux_bot.data(), fields.mp.at("u")->flux_top.data(),
                //fields.rhoref.data(), fields.rhorefh.data(),
                //fields.visc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.ijcells);

    diff_v<TF, Surface_model::Enabled>(
                fields.mt.at("v")->fld.data(),
                fields.sd.at("T12")->fld.data(), fields.sd.at("T22")->fld.data(), fields.sd.at("T23")->fld.data(),
                gd.z.data(), gd.zh.data(), 1./gd.dx, 1./gd.dy,
                fields.mp.at("v")->flux_bot.data(), fields.mp.at("v")->flux_top.data(),
                //fields.rhoref.data(), fields.rhorefh.data(),
                //fields.visc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.ijcells);

    diff_w<TF>(
                fields.mt.at("w")->fld.data(),
                fields.sd.at("T13")->fld.data(), fields.sd.at("T23")->fld.data(), fields.sd.at("T33")->fld.data(),
                fields.mp.at("u")->flux_bot.data(),fields.mp.at("v")->flux_bot.data(),
                gd.z.data(), gd.zh.data(), 1./gd.dx, 1./gd.dy,
                //fields.rhoref.data(), fields.rhorefh.data(),
                //fields.visc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.ijcells);

    for (auto it : fields.st)
        {
        diff_c<TF, Surface_model::Enabled>(
                it.second->fld.data(), fields.sp.at(it.first)->fld.data(),
                gd.dzi.data(), gd.dzhi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy),
                fields.sd.at("evisc")->fld.data(),
                fields.sp.at(it.first)->flux_bot.data(), fields.sp.at(it.first)->flux_top.data(),
                fields.rhoref.data(), fields.rhorefh.data(), tPr,
                fields.sp.at(it.first)->visc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.ijcells);
        }
    }

    molecular_diff_c<TF>(fields.mt.at("u")->fld.data(), fields.mp.at("u")->fld.data(), fields.visc,
               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend, gd.icells, gd.ijcells,
               gd.dx, gd.dy, gd.dzi.data(), gd.dzhi.data());

    molecular_diff_c<TF>(fields.mt.at("v")->fld.data(), fields.mp.at("v")->fld.data(), fields.visc,
               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend, gd.icells, gd.ijcells,
               gd.dx, gd.dy, gd.dzi.data(), gd.dzhi.data());

    molecular_diff_w<TF>(fields.mt.at("w")->fld.data(), fields.mp.at("w")->fld.data(), fields.visc,
               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend, gd.icells, gd.ijcells,
               gd.dx, gd.dy, gd.dzi.data(), gd.dzhi.data());
    
    /*for (auto it : fields.st) //Don't need this because molecular visc added to Smag eddy visc
        {
            molecular_diff_c<TF>(it.second->fld.data(), fields.sp.at(it.first)->fld.data(),fields.visc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend, gd.icells, gd.ijcells,
                gd.dx, gd.dy, gd.dzi.data(), gd.dzhi.data());
        }*/
    


    stats.calc_tend(*fields.mt.at("u"), tend_name);
    stats.calc_tend(*fields.mt.at("v"), tend_name);
    stats.calc_tend(*fields.mt.at("w"), tend_name);
    for (auto it : fields.st)
        stats.calc_tend(*it.second, tend_name);
}
#endif

template<typename TF>
void Diff_dnn<TF>::exec_viscosity(Thermo<TF>& thermo)
{
    auto& gd = grid.get_grid_data();
    auto grid_order = grid.get_spatial_order();
    auto buoy_tmp = fields.get_tmp();
    thermo.get_thermo_field(*buoy_tmp, "N2", false, false);
    
        if (boundary.get_switch() != "default")
    {
        const std::vector<TF>& z0m = boundary.get_z0m();

        // Calculate strain rate using MO for velocity gradients lowest level.
        const std::vector<TF>& dudz = boundary.get_dudz();
        const std::vector<TF>& dvdz = boundary.get_dvdz();

        calc_strain2<TF, Surface_model::Enabled>(
                fields.sd.at("evisc")->fld.data(),
                fields.mp.at("u")->fld.data(),
                fields.mp.at("v")->fld.data(),
                fields.mp.at("w")->fld.data(),
                dudz.data(),
                dvdz.data(),
                gd.z.data(),
                gd.dzi.data(),
                gd.dzhi.data(),
                1./gd.dx, 1./gd.dy,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells);
    }
    else
        // Calculate strain rate using resolved boundaries.
        calc_strain2<TF, Surface_model::Disabled>(
                fields.sd.at("evisc")->fld.data(),
                fields.mp.at("u")->fld.data(),
                fields.mp.at("v")->fld.data(),
                fields.mp.at("w")->fld.data(),
                nullptr, nullptr,
                gd.z.data(),
                gd.dzi.data(),
                gd.dzhi.data(),
                1./gd.dx, 1./gd.dy,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells);


    // Start with retrieving the stability information
    if (thermo.get_switch() == "0")
    {
        // Calculate eddy viscosity using MO at lowest model level
        if (boundary.get_switch() != "default")
        {
            const std::vector<TF>& z0m = boundary.get_z0m();

            calc_evisc_neutral<TF, Surface_model::Enabled>(
                    fields.sd.at("evisc")->fld.data(),
                    fields.mp.at("u")->fld.data(),
                    fields.mp.at("v")->fld.data(),
                    fields.mp.at("w")->fld.data(),
                    fields.mp.at("u")->flux_bot.data(),
                    fields.mp.at("v")->flux_bot.data(),
                    gd.z.data(), gd.dz.data(), gd.dzhi.data(), z0m.data(),
                    gd.dx, gd.dy, gd.zsize, this->cs, fields.visc,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.ijcells,
                    boundary_cyclic);
        }

        // Calculate eddy viscosity assuming resolved walls
        else
        {
            calc_evisc_neutral<TF, Surface_model::Disabled>(
                    fields.sd.at("evisc")->fld.data(),
                    fields.mp.at("u")->fld.data(),
                    fields.mp.at("v")->fld.data(),
                    fields.mp.at("w")->fld.data(),
                    fields.mp.at("u")->flux_bot.data(),
                    fields.mp.at("v")->flux_bot.data(),
                    gd.z.data(), gd.dz.data(), gd.dzhi.data(), nullptr,
                    gd.dx, gd.dy, gd.zsize, this->cs, fields.visc,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.ijcells,
                    boundary_cyclic);
        }
    }
    // assume buoyancy calculation is needed
    else
    {
        //auto buoy_tmp = fields.get_tmp();

        //thermo.get_thermo_field(*buoy_tmp, "N2", false, false);
        const std::vector<TF>& dbdz = boundary.get_dbdz();

        if (boundary.get_switch() != "default")
        {
            const std::vector<TF>& z0m = boundary.get_z0m();

            calc_evisc<TF, Surface_model::Enabled>(
                    fields.sd.at("evisc")->fld.data(),
                    fields.mp.at("u")->fld.data(),
                    fields.mp.at("v")->fld.data(),
                    fields.mp.at("w")->fld.data(),
                    buoy_tmp->fld.data(),
                    dbdz.data(),
                    gd.z.data(), gd.dz.data(),
                    gd.dzi.data(), z0m.data(),
                    gd.dx, gd.dy, this->cs, this->tPr,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.ijcells,
                    boundary_cyclic);
            /*
            calc_evisc_TKEbased<TF, Surface_model::Enabled>(
                    fields.sd.at("evisc")->fld.data(),
                    T11,T22, T33,
                    gd.dz.data(),
                    gd.dx, gd.dy, this->ce, //this->tPr,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.ijcells,
                    boundary_cyclic);
            */
            
        }
        else
        {
            calc_evisc<TF, Surface_model::Disabled>(
                    fields.sd.at("evisc")->fld.data(),
                    fields.mp.at("u")->fld.data(),
                    fields.mp.at("v")->fld.data(),
                    fields.mp.at("w")->fld.data(),
                    buoy_tmp->fld.data(),
                    nullptr,
                    gd.z.data(), gd.dz.data(),
                    gd.dzi.data(), nullptr,
                    gd.dx, gd.dy, this->cs, this->tPr,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.ijcells,
                    boundary_cyclic);
        }

        //fields.release_tmp(buoy_tmp);
      }

    destagger_u<TF, Surface_model::Enabled>(fields.sd.at("uc")->fld.data(), 
                    fields.mp.at("u")->fld.data(),
                    grid_order,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells,gd.ijcells,
                    boundary_cyclic);
        
    destagger_v<TF, Surface_model::Enabled>(fields.sd.at("vc")->fld.data(),
                    fields.mp.at("v")->fld.data(),
                    grid_order,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
        
    destagger_w<TF, Surface_model::Enabled>(fields.sd.at("wc")->fld.data(),
                    fields.mp.at("w")->fld.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    
    
    calc_TKEh<TF, Surface_model::Enabled>(fields.sd.at("TKEh")->fld.data(),
                    fields.sd.at("uc")->fld.data(),
                    fields.sd.at("vc")->fld.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    
    calc_TKEv<TF, Surface_model::Enabled>(fields.sd.at("TKEv")->fld.data(),
                    fields.sd.at("wc")->fld.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    
    calc_TPE<TF, Surface_model::Enabled>(fields.sd.at("TPE")->fld.data(),
                    fields.sp.at("b")->fld.data(),
                    buoy_tmp->fld.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);
    
    fields.release_tmp(buoy_tmp);
    
    Tau = calc_Tau<TF, Surface_model::Enabled>(
                    this->dnn,
                    fields.sd.at("uc")->fld.data(),
                    fields.sd.at("vc")->fld.data(),
                    fields.sd.at("wc")->fld.data(),
                    fields.sp.at("b")->fld.data(),
                    fields.sd.at("TKEh")->fld.data(),
                    fields.sd.at("TKEv")->fld.data(),
                    fields.sd.at("TPE")->fld.data(),
                    gd.dz.data(),
                    3,
                    gd.ncells,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        
    /*int ijk=gd.iend/2+ (gd.jend/2)*gd.icells + (gd.kend/3)*gd.ijcells;
    std::cout << Tau.index({ijk}) << std::endl; */
}

#ifndef USECUDA
template<typename TF>
void Diff_dnn<TF>::create_stats(Stats<TF>& stats)
{
    const std::string group_name = "default";

    // Add variables to the statistics
    if (stats.get_switch())
    {
        stats.add_profs(*fields.sd.at("evisc"), "z", {"mean", "2"}, group_name);
        stats.add_tendency(*fields.mt.at("u"), "z", tend_name, tend_longname);
        stats.add_tendency(*fields.mt.at("v"), "z", tend_name, tend_longname);
        stats.add_tendency(*fields.mt.at("w"), "zh", tend_name, tend_longname);

        for (auto it : fields.st)
            stats.add_tendency(*it.second, "z", tend_name, tend_longname);
    }
}
#endif

template<typename TF>
void Diff_dnn<TF>::exec_stats(Stats<TF>& stats)
{
    const TF no_offset = 0.;
    const TF no_threshold = 0.;
    stats.calc_stats("evisc", *fields.sd.at("evisc"), no_offset, no_threshold);
}

template<typename TF>
void Diff_dnn<TF>::diff_flux(Field3d<TF>& restrict out, const Field3d<TF>& restrict fld_in)
{
    auto& gd = grid.get_grid_data();

    if (boundary.get_switch() != "default")
    {
        // Calculate the boundary fluxes.
        calc_diff_flux_bc(out.fld.data(), fld_in.flux_bot.data(), gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.icells, gd.ijcells);
        calc_diff_flux_bc(out.fld.data(), fld_in.flux_top.data(), gd.istart, gd.iend, gd.jstart, gd.jend, gd.kend  , gd.icells, gd.ijcells);

        // Calculate the interior.
        if (fld_in.loc[0] == 1)
            calc_diff_flux_u<TF, Surface_model::Enabled>(
                    out.fld.data(), fld_in.fld.data(), fields.mp.at("w")->fld.data(), fields.sd.at("evisc")->fld.data(),
                    gd.dxi, gd.dzhi.data(),
                    fields.visc,
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        else if (fld_in.loc[1] == 1)
            calc_diff_flux_v<TF, Surface_model::Enabled>(
                    out.fld.data(), fld_in.fld.data(), fields.mp.at("w")->fld.data(), fields.sd.at("evisc")->fld.data(),
                    gd.dyi, gd.dzhi.data(),
                    fields.visc,
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        else
            calc_diff_flux_c<TF, Surface_model::Enabled>(
                    out.fld.data(), fld_in.fld.data(), fields.sd.at("evisc")->fld.data(),
                    gd.dzhi.data(),
                    tPr, fld_in.visc,
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
    }
    else
    {
        // Include the wall.
        if (fld_in.loc[0] == 1)
            calc_diff_flux_u<TF, Surface_model::Disabled>(
                    out.fld.data(), fld_in.fld.data(), fields.mp.at("w")->fld.data(), fields.sd.at("evisc")->fld.data(),
                    gd.dxi, gd.dzhi.data(),
                    fields.visc,
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        else if (fld_in.loc[1] == 1)
            calc_diff_flux_v<TF, Surface_model::Disabled>(
                    out.fld.data(), fld_in.fld.data(), fields.mp.at("w")->fld.data(), fields.sd.at("evisc")->fld.data(),
                    gd.dyi, gd.dzhi.data(),
                    fields.visc,
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        else
            calc_diff_flux_c<TF, Surface_model::Disabled>(
                    out.fld.data(), fld_in.fld.data(), fields.sd.at("evisc")->fld.data(),
                    gd.dzhi.data(),
                    tPr, fld_in.visc,
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
    }
}
template class Diff_dnn<double>;
template class Diff_dnn<float>;

/*
    template<typename TF, Surface_model surface_model>
    void calc_evisc_TKEbased(
            TF* const restrict evisc,
            const TF* const restrict T11,
            const TF* const restrict T22,
            const TF* const restrict T33,
            const TF* const restrict dz,
            const TF dx, const TF dy,
            const TF ce, //const TF tPr,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells, const int ijcells,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int jj = icells;
        const int kk = ijcells;
        
        for (int k=kstart+1; k<kend; ++k)    
        {
            const TF lgrid = std::pow(dx*dy*dz[k], TF(1./3.));
            for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ij  = i + j*jj;
                        const int ijk = i + j*jj + k*kk;

                         // Create a vector of inputs.
                         evisc[ijk] = ce*lgrid*std::pow(TF(0.5)*(T11[ijk]+T22[ijk]+T33[ijk]),0.5); // Don't divide by tPr here, done later
                    }
        }
        boundary_cyclic.exec(evisc);
    }
*/
