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

    
    template <typename TF, Surface_model surface_model>
    void destagger_u(
            TF* const restrict uc,
            const TF* const restrict u,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        
        const int ii = 1;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend-ii; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    uc[ijk] = TF(0.5)*(u[ijk+ii]+u[ijk]);
                }
    }
    
    template <typename TF, Surface_model surface_model>
    void destagger_v(
            TF* const restrict vc,
            const TF* const restrict v,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        
        const int ii = 1;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend-jj; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    vc[ijk] = TF(0.5)*(v[ijk+jj]+v[ijk]);
                }
    }
    
    template <typename TF, Surface_model surface_model>
    void destagger_w(
            TF* const restrict wc,
            const TF* const restrict w,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        
        const int ii = 1;

        for (int k=kstart; k<kend-kk; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    wc[ijk] = TF(0.5)*(w[ijk+kk]+w[ijk]);
                }
    }
    
    template <typename TF, Surface_model surface_model>
    void calc_TKEh(
            TF* const restrict TKEh,
            const TF* const restrict uc,
            const TF* const restrict vc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        const int ii = 1;
        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;
        // If the wall isn't resolved, calculate du/dz and dv/dz at lowest grid height using MO
        if (surface_model == Surface_model::Enabled)
        {
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;
                    
                    TKEh[ijk] = TKEh[ijk] = TF(0.25)*(
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
                }
        }

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
                    TKEh[ijk] += Constants::dsmall;
                }
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
                    
                    ut[ijk] +=
                            // -dTau11/dx
                            -dxi*(T11[ijk+ii]-T11[ijk])
                            // -dTau12/dy
                            -TF(0.25)*dyi*(T12[ijk+jj]+T12[ijk+ii+jj]-T12[ijk-jj]-T12[ijk+ii-jj]) 
                             // -dTau13/dz
                            -TF(0.5)*(T13[ijk+kk]+T13[ijk+ii+kk]-fluxbot[ijk]-fluxbot[ijk+ii])/(z[kstart+1]-zh[kstart]);
                }

            // top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kend-1)*kk;
                    
                    ut[ijk] +=
                            // -dTau11/dx
                            -dxi*(T11[ijk+ii]-T11[ijk])
                            // -dTau12/dy
                            -TF(0.25)*dyi*(T12[ijk+jj]+T12[ijk+ii+jj]-T12[ijk-jj]-T12[ijk+ii-jj]) 
                             // -dTau13/dz
                            -TF(0.5)*(fluxtop[ijk+kk]+fluxtop[ijk+ii+kk]-T13[ijk]-T13[ijk+ii])/(zh[kend]-z[kend-1]);

                  }
        }

        for (int k=kstart+k_offset; k<kend-k_offset; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    ut[ijk] +=
                             // -dTau11/dx
                            -dxi*(T11[ijk+ii]-T11[ijk])
                            // -dTau12/dy
                            -TF(0.25)*dyi*(T12[ijk+jj]+T12[ijk+ii+jj]-T12[ijk-jj]-T12[ijk+ii-jj]) 
                             // -dTau13/dz
                            -TF(0.5)*(z[k+1]-z[k-1])*(T13[ijk+kk]+T13[ijk+ii+kk]-T12[ijk-kk]-T13[ijk+ii-kk]); 
                }
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
                    
                    vt[ijk] +=
                            // -dTau21/dx
                            -TF(0.25)*dxi*(T12[ijk+ii]+T12[ijk+ii+jj]-T12[ijk-ii]-T12[ijk-ii+jj])
                            // -dTau22/dy
                            -dyi*(T22[ijk+jj]-T22[ijk]) 
                             // -dTau23/dz
                            -TF(0.5)*(T23[ijk+kk]+T23[ijk+jj+kk]-fluxbot[ijk]-fluxbot[ijk+jj])/(z[kstart+1]-zh[kstart]);
                }

            // top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kend-1)*kk;
                    
                    vt[ijk] +=
                           // -dTau21/dx
                            -TF(0.25)*dxi*(T12[ijk+ii]+T12[ijk+ii+jj]-T12[ijk-ii]-T12[ijk-ii+jj])
                            // -dTau22/dy
                            -dyi*(T22[ijk+jj]-T22[ijk]) 
                             // -dTau23/dz
                            -TF(0.5)*(fluxtop[ijk+kk]+fluxtop[ijk+jj+kk]-T23[ijk]-T23[ijk+jj])/(zh[kend]-z[kend-1]);

                  }
        }

        for (int k=kstart+k_offset; k<kend-k_offset; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    vt[ijk] +=
                            // -dTau21/dx
                            -TF(0.25)*dxi*(T12[ijk+ii]+T12[ijk+ii+jj]-T12[ijk-ii]-T12[ijk-ii+jj])
                            // -dTau22/dy
                            -dyi*(T22[ijk+jj]-T22[ijk]) 
                             // -dTau23/dz
                            -TF(0.5)*(T23[ijk+kk]+T23[ijk+jj+kk]-T23[ijk-kk]-T23[ijk+jj-kk])/(z[k+1]-z[k-1]); 
                }
    }

    template <typename TF>
    void diff_w(
            TF* const restrict wt,
            const TF* const restrict T13,
            const TF* const restrict T23,
            const TF* const restrict T33,
            const TF* const restrict z,
            const TF* const restrict zh,
            const TF dxi, const TF dyi,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        const int ii = 1;

        if (1)
        {
            // first above bottom boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {    
                    
                    const int ijk = i + j*jj + (kstart+1)*kk;
                    wt[ijk] +=
                            // -dTau31/dx
                            -TF(0.25)*dxi*(T13[ijk+ii]+T13[ijk+ii+kk]-T13[ijk-ii]-T13[ijk-ii+kk])
                            // -dTau32/dy
                            -TF(0.25)*dxi*(T23[ijk+jj]+T23[ijk+jj+kk]-T23[ijk-jj]-T23[ijk-jj+kk])
                             // -dTau33/dz
                            -T33[ijk]/(z[kstart+1]-zh[kstart]);
                }

            // first below top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {

                    const int ijk = i + j*jj + (kend-1)*kk;
                    
                    wt[ijk] +=
                           // -dTau31/dx
                            -TF(0.25)*dxi*(T13[ijk+ii]+T13[ijk+ii+kk]-T13[ijk-ii]-T13[ijk-ii+kk])
                            // -dTau32/dy
                            -TF(0.25)*dxi*(T23[ijk+jj]+T23[ijk+jj+kk]-T23[ijk-jj]-T23[ijk-jj+kk])
                             // -dTau33/dz
                            -T33[ijk]/(zh[kend]-z[kend-1]);

                  }
        }

        for (int k=kstart+1; k<kend-1; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    wt[ijk] +=
                            // -dTau31/dx
                            -TF(0.25)*dxi*(T13[ijk+ii]+T13[ijk+ii+kk]-T13[ijk-ii]-T13[ijk-ii+kk])
                            // -dTau32/dy
                            -TF(0.25)*dxi*(T23[ijk+jj]+T23[ijk+jj+kk]-T23[ijk-jj]-T23[ijk-jj+kk])
                             // -dTau33/dz
                            -(T33[ijk+kk]-T33[ijk-kk])/(z[k+1]-z[k-1]); 
                }
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

  template<typename TF, Surface_model surface_model>
    void calc_evisc(
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

    const std::string group_name = "default";

    fields.init_diagnostic_field("evisc", "Eddy viscosity", "m2 s-1", group_name, gd.sloc);
    fields.init_diagnostic_field("T11", "Turbulent flux of u_1 mom'm in x_1 direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T12", "Turbulent flux of u_1(2) mom'm in x_2(1) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T13", "Turbulent flux of u_1(3) mom'm in x_3(1) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T22", "Turbulent flux of u_2 mom'm in x_2 direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T23", "Turbulent flux of u_2(3) mom'm in x_3(2) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T33", "Turbulent flux of u_3 mom'm in x_3 direction", "m2 s-2", group_name, gd.sloc);

    if (grid.get_spatial_order() != Grid_order::Second)
        throw std::runtime_error("Diff_dnn only runs with second order grids");
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

#ifndef USECUDA
template<typename TF>
void Diff_dnn<TF>::exec(Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();

    if (boundary.get_switch() != "default")
    {
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
    


    stats.calc_tend(*fields.mt.at("u"), tend_name);
    stats.calc_tend(*fields.mt.at("v"), tend_name);
    stats.calc_tend(*fields.mt.at("w"), tend_name);
    for (auto it : fields.st)
        stats.calc_tend(*it.second, tend_name);
}

template<typename TF>
void Diff_dnn<TF>::exec_viscosity(Thermo<TF>& thermo)
{
           // Store the buoyancy flux in tmp1
        auto& gd = grid.get_grid_data();
        auto buoy_tmp = fields.get_tmp();
        thermo.get_thermo_field(*buoy_tmp, "N2", false, false);
        const std::vector<TF>& dbdz = boundary.get_dbdz();

        if (boundary.get_switch() != "default")
        {
            const std::vector<TF>& z0m = boundary.get_z0m();

                                   
            //calc_tau<TF, Surface_model::Enabled>(output);
            //std::vector<TF> Tau[gd.ntot];  
            //std::fill_n(Tau, gd.ntot, TF(0.00001));// torch::ones({gd.ntot, 6})*TF(0.00001);
            //Tau = fields.visc
            
            at::Tensor Tau = torch::rand({gd.ntot, 6}).to(torch::kDouble);
            //std::cout << Tau << std::endl;

            TF* T11 = Tau.slice(/*dim=*/1, /*start=*/0, /*end=*/0).data_ptr<TF>();
            TF* T12 = Tau.slice(/*dim=*/1, /*start=*/1, /*end=*/1).data_ptr<TF>();
            TF* T13 = Tau.slice(/*dim=*/1, /*start=*/2, /*end=*/2).data_ptr<TF>();
            TF* T22 = Tau.slice(/*dim=*/1, /*start=*/3, /*end=*/3).data_ptr<TF>();
            TF* T23 = Tau.slice(/*dim=*/1, /*start=*/4, /*end=*/4).data_ptr<TF>();
            TF* T33 = Tau.slice(/*dim=*/1, /*start=*/5, /*end=*/5).data_ptr<TF>(); 
            
            boundary_cyclic.exec(T11);
            boundary_cyclic.exec(T12);
            boundary_cyclic.exec(T13);
            boundary_cyclic.exec(T22);
            boundary_cyclic.exec(T23);
            boundary_cyclic.exec(T33);
            
            calc_evisc<TF, Surface_model::Enabled>(
                    fields.sd.at("evisc")->fld.data(),
                    T11,T22, T33,
                    gd.dz.data(),
                    gd.dx, gd.dy, this->ce, //this->tPr,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.ijcells,
                    boundary_cyclic);
                       
            fields.release_tmp(buoy_tmp);
            //fields.release_tmp(tmp);
    }
}
#endif

/*
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

  
*/