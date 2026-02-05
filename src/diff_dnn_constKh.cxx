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

#include "diff_dnn_constKh.h"

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
                    uc[ijk] = TF(0.5)*(u[ijk+ii]+u[ijk]);
                }
        
        boundary_cyclic.exec(uc);
    }
    
    template <typename TF, Surface_model surface_model>
    void destagger_v(
            TF* const restrict vc,
            const TF* const restrict v,
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
    void calc_S2c(
            TF* const restrict S2c,
            const TF* const restrict uc,
            const TF* const restrict vc,    
            const TF* const restrict wc,
            const TF dxi, const TF dyi,
            const TF* z,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int ii = 1;
                
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                for (int k=kstart+1; k<kend; ++k)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    const TF d11 = dxi*(uc[ijk+ii] - uc[ijk-ii])/2.0;
                    const TF d12 = dyi*(uc[ijk+jj] - uc[ijk-jj])/2.0;
                    const TF d13 = (uc[ijk+kk] - uc[ijk-kk])/(z[k+1]-z[k-1]);

                    const TF d21 = dxi*(vc[ijk+ii] - vc[ijk-ii])/2.0;
                    const TF d22 = dyi*(vc[ijk+jj] - vc[ijk-jj])/2.0;
                    const TF d23 = (vc[ijk+kk] - vc[ijk-kk])/(z[k+1]-z[k-1]);

                    const TF d31 = dxi*(wc[ijk+ii] - wc[ijk-ii])/2.0;
                    const TF d32 = dyi*(wc[ijk+jj] - wc[ijk-jj])/2.0;
                    const TF d33 = (wc[ijk+kk] - wc[ijk-kk])/(z[k+1]-z[k-1]);

                    S2c[ijk] = 2*(d11*d11 + d22*d22 + d33*d33) + 2*(d12*d21 + d13*d31 + d23*d32) + d12*d12 + d21*d21 + d13*d13 + d31*d31 + d23*d23 + d32*d32;
                }
            }
        boundary_cyclic.exec(S2c);
    }

    template <typename TF, Surface_model surface_model>
    void calc_N2c(
            TF* const restrict N2c,
            const TF* const restrict b,
            const TF* z,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int ii = 1;
                
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                for (int k=kstart+1; k<kend; ++k)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    N2c[ijk] = (b[ijk+kk] - b[ijk-kk])/(z[k+1]-z[k-1]);
                }
            }
        boundary_cyclic.exec(N2c);
    }

    template <typename TF, Surface_model surface_model>
    at::Tensor calc_Tau(
            torch::jit::script::Module dnn,
            const TF* const restrict uc,
            const TF* const restrict vc,    
            const TF* const restrict wc,
            const TF* const restrict N2,
            const TF* const restrict S2,
            const TF ivel_scale,
            const TF stress_scale,
            const TF Ri_char,
            const bool swdeviatoric,
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
        
        at::Tensor u_inputs = torch::zeros({nbatch, 3*nv, nh, nh}); // Number of input velocity components, 3, is fixed: u,v,w
        at::Tensor Ri_inputs = torch::zeros({nbatch, 1, 1, 1}); // Awkward size until R2conv on 1x1 -> new Linear in escnn       

        for (int k=kstart+k_offset; k<kendBL-1; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const int ijkbatch = i-istart + (j-jstart)*jjbatch + (k-kstart-k_offset)*kkbatch;

                    TF ubar = 0;
                    TF vbar = 0;
                    TF wbar = 0;
                    for (int ix=-ih; ix<=ih; ix++)
                        for (int iy=-ih; iy<=ih; iy++)
                            for (int iz=-iv; iz<=iv; iz++)
                            {
                                ubar += uc[ijk+ix*ii+iy*jj+iz*kk];
                                vbar += vc[ijk+ix*ii+iy*jj+iz*kk];
                                wbar += wc[ijk+ix*ii+iy*jj+iz*kk];
                             }
                    ubar=ubar/nbox;
                    vbar=vbar/nbox;
                    wbar=wbar/nbox;
                    
                    for (int iz=-iv; iz<=iv; iz++)
                        for (int ix=-ih; ix<=ih; ++ix)
                                for (int iy=-ih; iy<=ih; ++iy)
                                {                                    
                                    u_inputs.index_put_({ijkbatch, 2*(iz+iv),ih+ix,ih+iy}, (uc[ijk+ix*ii+iy*jj+iz*kk]-ubar)*ivel_scale);
                                    u_inputs.index_put_({ijkbatch, 2*(iz+iv)+1,ih+ix,ih+iy},(vc[ijk+ix*ii+iy*jj+iz*kk]-vbar)*ivel_scale);
                                    u_inputs.index_put_({ijkbatch, 2*nv+(iz+iv),ih+ix,ih+iy},(wc[ijk+ix*ii+iy*jj+iz*kk]-wbar)*ivel_scale);
                                    
                                }
                    /*if (ijkbatch==4){std::cout << u_inputs.slice(0, 0, 4) << std::endl;}
                    if (ijk==(iend/2+ (jend/2)*jj + (kend/3)*kk)){std::cout << u_inputs.slice(0, ijk, ijk+1) << std::endl;}*/

                    Ri_inputs.index_put_({ijkbatch, 0, 0, 0},(N2[ijk] - Ri_char*S2[ijk])/(N2[ijk]+Ri_char*S2[ijk]));

                    /*if (ijkbatch==4){std::cout << Ri_inputs.slice(0, 0, 4) << std::endl;}
                    if (ijk==(iend/2+ (jend/2)*jj + (kend/3)*kk)){std::cout << Ri_inputs.slice(0, ijk, ijk+1) << std::endl;}*/
                }
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(u_inputs);
        inputs.push_back(Ri_inputs);

        at::Tensor Tau = dnn.forward(inputs).toTensor();

        for (int k=kstart+k_offset; k<kendBL-1; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const int ijkbatch = i-istart + (j-jstart)*jjbatch + (k-kstart-k_offset)*kkbatch;
                                        
                    if(swdeviatoric)
                    {
                        const auto third_trace = TF(1.0/3.0)*(Tau.index({ijkbatch, 0}) + Tau.index({ijkbatch, 3}) + Tau.index({ijkbatch, 5}));
                        
                        /*if (ijkbatch==4) 
                        {std::cout << Tau.slice(0, 0, 4) << std::endl;}
                        if (ijk== (iend/2+ (jend/2)*jj + (kend/3)*kk))
                        {std::cout << Tau.slice(0, ijk,ijk+1) << std::endl;}*/
                        Tau.index_put_({ijkbatch, 0}, (Tau.index({ijkbatch, 0})-third_trace) * stress_scale );
                        Tau.index_put_({ijkbatch, 1},  Tau.index({ijkbatch, 1}) * stress_scale );
                        Tau.index_put_({ijkbatch, 2},  Tau.index({ijkbatch, 2}) * stress_scale );
                        Tau.index_put_({ijkbatch, 3}, (Tau.index({ijkbatch, 3})-third_trace) * stress_scale );
                        Tau.index_put_({ijkbatch, 4},  Tau.index({ijkbatch, 4}) * stress_scale );
                        Tau.index_put_({ijkbatch, 5}, (Tau.index({ijkbatch, 5})-third_trace) * stress_scale );
                        /*if (ijkbatch==4) 
                        {std::cout << Tau.slice(0, 0, 4) << std::endl;}
                        if (ijk==(iend/2+ (jend/2)*jj + (kend/3)*kk)) 
                        {std::cout << Tau.slice(0, ijk,ijk+1) << std::endl;} */
                    }
                    else
                    {
                        /*if (ijkbatch==4) 
                        {std::cout << Tau.slice(0, 0, 4) << std::endl;}
                        if (ijk== (iend/2+ (jend/2)*jj + (kend/3)*kk)) 
                        {std::cout << Tau.slice(0, ijk,ijk+1) << std::endl;}*/
                        Tau.index_put_({ijkbatch, 0},  Tau.index({ijkbatch, 0}) * stress_scale );
                        Tau.index_put_({ijkbatch, 1},  Tau.index({ijkbatch, 1}) * stress_scale );
                        Tau.index_put_({ijkbatch, 2},  Tau.index({ijkbatch, 2}) * stress_scale );
                        Tau.index_put_({ijkbatch, 3},  Tau.index({ijkbatch, 3}) * stress_scale );
                        Tau.index_put_({ijkbatch, 4},  Tau.index({ijkbatch, 4}) * stress_scale );
                        Tau.index_put_({ijkbatch, 5},  Tau.index({ijkbatch, 5}) * stress_scale );
                        /*if (ijkbatch==04) 
                        {std::cout << Tau.slice(0, 0, 4) << std::endl;}
                        if (ijk==(iend/2+ (jend/2)*jj + (kend/3)*kk)) 
                        {std::cout << Tau.slice(0, ijk,ijk+1) << std::endl;} */
                    }
                                       
                    
                }
                
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
                flux_fld[ijk+kk] = flux_fld[ijk]; // if fluxtop is known somehow, change this to interpolation
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
                            // -dTau11/dx 
                            -dxi*(T11[ijk+ii]-T11[ijk])
                            // -dTau12/dy 
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

        for (int k=kstart+1; k<kendBL; ++k)
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

        // DNN turned off above boundary layer
             /*for (int k=kendBL; k<kend-1; ++k)   ...
             const int ijk = i + j*jj + (kend-1)*kk; */
    }

    template <typename TF, Surface_model surface_model>
    void constK_diff_c(
            TF* const restrict at,
            const TF* const restrict a,
            const TF* const restrict dzi,
            const TF* const restrict dzhi,
            const TF dxidxi, const TF dyidyi,
            const TF visc,
            const TF* const restrict fluxbot,
            const TF* const restrict fluxtop,
            const TF* const restrict rhoref,
            const TF* const restrict rhorefh,
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
                    
                    at[ijk] +=
                             + ( visc*(a[ijk+ii]-a[ijk   ])
                               - visc*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( visc*(a[ijk+jj]-a[ijk   ])
                               - visc*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + ( rhorefh[kstart+1] * visc*(a[ijk+kk]-a[ijk   ])*dzhi[kstart+1]
                               + rhorefh[kstart  ] * fluxbot[ij] ) / rhoref[kstart] * dzi[kstart];
                }

            // top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kend-1)*kk;
                    
                    at[ijk] +=
                             + ( visc*(a[ijk+ii]-a[ijk   ])
                               - visc*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( visc*(a[ijk+jj]-a[ijk   ])
                               - visc*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + (-rhorefh[kend  ] * fluxtop[ij]
                               - rhorefh[kend-1] * visc*(a[ijk   ]-a[ijk-kk])*dzhi[kend-1] ) / rhoref[kend-1] * dzi[kend-1];
                }
        }

        for (int k=kstart+k_offset; k<kend-k_offset; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    
                    at[ijk] +=
                             + ( visc*(a[ijk+ii]-a[ijk   ])
                               - visc*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( visc*(a[ijk+jj]-a[ijk   ])
                               - visc*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + ( rhorefh[k+1] * visc*(a[ijk+kk]-a[ijk   ])*dzhi[k+1]
                               - rhorefh[k  ] * visc*(a[ijk   ]-a[ijk-kk])*dzhi[k]  ) / rhoref[k] * dzi[k];
                }
    }

    template<typename TF>
    void molecular_diff_c(TF* restrict at, const TF* restrict a, const TF visc,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk, const TF dx, const TF dy, const TF* restrict dzi, const TF* restrict dzhi)
    {
        const int ii = 1;
        const double dxidxi = 1/(dx*dx);
        const double dyidyi = 1/(dy*dy);

        // bottom boundary
        for (int j=jstart; j<jend; j++)
            #pragma ivdep
            for (int i=istart; i<iend; i++)
            {
                const int ijk = i + j*jj + kstart*kk;
                
                at[ijk] += visc * (
                        + ( (a[ijk+ii] - a[ijk   ])
                          - (a[ijk   ] - a[ijk-ii]) ) * dxidxi
                        + ( (a[ijk+jj] - a[ijk   ])
                          - (a[ijk   ] - a[ijk-jj]) ) * dyidyi
                        + ( (a[ijk+2*kk] - a[ijk+kk]) * dzhi[kstart+2]
                          - (a[ijk+kk ] - a[ijk]) * dzhi[kstart+1]   ) * dzi[kstart+1] );
            }

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

        //top boundary handled above as long as zero Neumann
        
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
    
    template<typename TF>
    TF calc_dnmul(
            const TF visc,
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
                    dnmul = std::max(dnmul, std::abs(visc*tPrfac_i*(dxidxi + dyidyi + dzi[k]*dzi[k])));
                }

        return dnmul;
    }

    template <typename TF, Surface_model surface_model>
    void calc_diff_flux_c(
            TF* const restrict out,
            const TF* const restrict data,
            const TF visc,
            const TF* const restrict dzhi,
            const TF tPr,
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

                    out[ijk] = - visc*(data[ijk] - data[ijk-kk])*dzhi[k];
                }
        }
    }

    template <typename TF, Surface_model surface_model>
    void calc_diff_flux_u(
            TF* const restrict out,
            const TF* const restrict data,
            const TF* const restrict w,
            const TF visc,
            const TF dxi, const TF* const dzhi,
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
                    
                    out[ijk] = - visc*( (data[ijk]-data[ijk-ijcells])*dzhi[k] + (w[ijk]-w[ijk-ii])*dxi );
                }
        }
    }

    template <typename TF, Surface_model surface_model>
    void calc_diff_flux_v(
            TF* const restrict out,
            const TF* const restrict data,
            const TF* const restrict w,
            const TF visc,
            const TF dyi, const TF* const dzhi,
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
                        
                        out[ijk] = - visc*( (data[ijk]-data[ijk-ijcells])*dzhi[k] + (w[ijk]-w[ijk-icells])*dyi );
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
Diff_dnn_constKh<TF>::Diff_dnn_constKh(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Boundary<TF>& boundaryin, Input& inputin) :
    Diff<TF>(masterin, gridin, fieldsin, boundaryin, inputin),
    boundary_cyclic(master, grid),
    field3d_operators(master, grid, fields)
{
    auto& gd = grid.get_grid_data();
    dnmax = inputin.get_item<TF>("diff", "dnmax", "", 0.4  );
    cs    = inputin.get_item<TF>("diff", "cs"   , "", 0.23 );
    tPr   = inputin.get_item<TF>("diff", "tPr"  , "", 1.);
    Re    = inputin.get_item<TF>("diff", "Re"  , "", 40000.0);
    Ug    = inputin.get_item<TF>("diff", "Ug"  , "", 0.05);
    Ri_char = inputin.get_item<TF>("diff", "Ri_char"  , "", 5.0);
    swdeviatoric   = inputin.get_item<TF>("diff", "swdeviatoric", "", true);
    ivel_scale =  std::sqrt(Re)/Ug;
    stress_scale = Ug*Ug/Re;
    swdeviatoric   = inputin.get_item<TF>("diff", "swdeviatoric", "", true);
    dnnpath = inputin.get_item<std::string>("diff", "dnnpath"  , "", "C4_midGridReInterp_local_4x1026Re900_4x3078Re2700_1.pt");
    try {dnn = torch::jit::load(dnnpath);}
    catch (const c10::Error& e) {std::cerr << "error loading the deep neural network\n";}
        
    const std::string group_name = "default";

    fields.init_diagnostic_field("T11", "Turbulent flux of u_1 mom'm in x_1 direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T12", "Turbulent flux of u_1(2) mom'm in x_2(1) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T13", "Turbulent flux of u_1(3) mom'm in x_3(1) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T22", "Turbulent flux of u_2 mom'm in x_2 direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T23", "Turbulent flux of u_2(3) mom'm in x_3(2) direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("T33", "Turbulent flux of u_3 mom'm in x_3 direction", "m2 s-2", group_name, gd.sloc);
    fields.init_diagnostic_field("uc", "Destaggered (centered) u velocity", "m s-1", group_name, gd.sloc);
    fields.init_diagnostic_field("vc", "Destaggered (centered) v velocity", "m s-1", group_name, gd.sloc);
    fields.init_diagnostic_field("wc", "Destaggered (centered) w velocity", "m s-1", group_name, gd.sloc);
    fields.init_diagnostic_field("S2c", "Magnitude of strain rate from centered DNN inputs", "", group_name, gd.sloc);
    fields.init_diagnostic_field("N2c", "Buoyancy frequency for DNN", "", group_name, gd.sloc);

/*    if (grid.get_spatial_order() != Grid_order::Second)
        throw std::runtime_error("Diff_dnn_constKh only runs with second order grids");*/
}

template<typename TF>
Diff_dnn_constKh<TF>::~Diff_dnn_constKh()
{
}

template<typename TF>
void Diff_dnn_constKh<TF>::init()
{
    boundary_cyclic.init();
}

template<typename TF>
Diffusion_type Diff_dnn_constKh<TF>::get_switch() const
{
    return swdiff;
}

#ifndef USECUDA
template<typename TF>
unsigned long Diff_dnn_constKh<TF>::get_time_limit(const unsigned long idt, const double dt)
{
    auto& gd = grid.get_grid_data();

    double dnmul = calc_dnmul<TF>(
        fields.visc,
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
double Diff_dnn_constKh<TF>::get_dn(const double dt)
{
    auto& gd = grid.get_grid_data();

    double dnmul = calc_dnmul<TF>(
        fields.visc,
        gd.dzi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy), tPr,
        gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
        gd.icells, gd.ijcells);
    master.max(&dnmul, 1);

    return dnmul*dt;
}
#endif

template<typename TF>
void Diff_dnn_constKh<TF>::create(Stats<TF>& stats)
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
void Diff_dnn_constKh<TF>::exec(Stats<TF>& stats)
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
        constK_diff_c<TF, Surface_model::Enabled>(
                it.second->fld.data(), fields.sp.at(it.first)->fld.data(),
                gd.dzi.data(), gd.dzhi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy),
                fields.sp.at(it.first)->visc,
                fields.sp.at(it.first)->flux_bot.data(), fields.sp.at(it.first)->flux_top.data(),
                fields.rhoref.data(), fields.rhorefh.data(),
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

    stats.calc_tend(*fields.mt.at("u"), tend_name);
    stats.calc_tend(*fields.mt.at("v"), tend_name);
    stats.calc_tend(*fields.mt.at("w"), tend_name);
    for (auto it : fields.st)
        stats.calc_tend(*it.second, tend_name);
}
#endif

template<typename TF>
void Diff_dnn_constKh<TF>::exec_viscosity(Thermo<TF>& thermo)
{
    auto& gd = grid.get_grid_data();
    auto grid_order = grid.get_spatial_order();

    destagger_u<TF, Surface_model::Enabled>(fields.sd.at("uc")->fld.data(),
                    fields.mp.at("u")->fld.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells,gd.ijcells,
                    boundary_cyclic);
        
    destagger_v<TF, Surface_model::Enabled>(fields.sd.at("vc")->fld.data(),
                    fields.mp.at("v")->fld.data(),
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

     calc_S2c<TF, Surface_model::Enabled>(fields.sd.at("S2c")->fld.data(),
                    fields.sd.at("uc")->fld.data(),
                    fields.sd.at("vc")->fld.data(),
                    fields.sd.at("wc")->fld.data(),
                    gd.dxi, gd.dyi,
                    gd.z.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells,
                    boundary_cyclic);

    calc_N2c<TF, Surface_model::Enabled>(fields.sd.at("N2c")->fld.data(), 
                    fields.sp.at("b")->fld.data(),
                    gd.z.data(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells,gd.ijcells,
                    boundary_cyclic);

    // auto buoy_tmp = fields.get_tmp();
    // thermo.get_thermo_field(*buoy_tmp, "N2", false, false);
    // const std::vector<TF>& dbdz = boundary.get_dbdz();
    
    Tau = calc_Tau<TF, Surface_model::Enabled>(
                    this->dnn,
                    fields.sd.at("uc")->fld.data(),
                    fields.sd.at("vc")->fld.data(),
                    fields.sd.at("wc")->fld.data(),
                    // buoy_tmp->fld.data(),
                    fields.sd.at("N2c")->fld.data(),
                    fields.sd.at("S2c")->fld.data(),
                    this->ivel_scale,
                    this->stress_scale,
                    this->Ri_char,
                    this->swdeviatoric,
                    3,
                    gd.ncells,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        
    /*int ijk=gd.iend/2+ (gd.jend/2)*gd.icells + (gd.kend/3)*gd.ijcells;
    std::cout << Tau.index({ijk}) << std::endl; */

    // fields.release_tmp(buoy_tmp);
}

#ifndef USECUDA
template<typename TF>
void Diff_dnn_constKh<TF>::create_stats(Stats<TF>& stats)
{
    const std::string group_name = "default";

    // Add variables to the statistics
    if (stats.get_switch())
    {
        //stats.add_profs(*fields.sd.at("evisc"), "z", {"mean", "2"}, group_name);
        stats.add_tendency(*fields.mt.at("u"), "z", tend_name, tend_longname);
        stats.add_tendency(*fields.mt.at("v"), "z", tend_name, tend_longname);
        stats.add_tendency(*fields.mt.at("w"), "zh", tend_name, tend_longname);

        for (auto it : fields.st)
            stats.add_tendency(*it.second, "z", tend_name, tend_longname);
    }
}
#endif

template<typename TF>
void Diff_dnn_constKh<TF>::exec_stats(Stats<TF>& stats)
{
    const TF no_offset = 0.;
    const TF no_threshold = 0.;
    //stats.calc_stats("evisc", *fields.sd.at("evisc"), no_offset, no_threshold);
}

template<typename TF>
void Diff_dnn_constKh<TF>::diff_flux(Field3d<TF>& restrict out, const Field3d<TF>& restrict fld_in)
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
                    out.fld.data(), fld_in.fld.data(), fields.mp.at("w")->fld.data(), fields.visc,
                    gd.dxi, gd.dzhi.data(),
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        else if (fld_in.loc[1] == 1)
            calc_diff_flux_v<TF, Surface_model::Enabled>(
                    out.fld.data(), fld_in.fld.data(), fields.mp.at("w")->fld.data(), fields.visc,
                    gd.dyi, gd.dzhi.data(),
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        else
            calc_diff_flux_c<TF, Surface_model::Enabled>(
                    out.fld.data(), fld_in.fld.data(), fld_in.visc,
                    gd.dzhi.data(),
                    tPr, 
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
    }

}
template class Diff_dnn_constKh<double>;
template class Diff_dnn_constKh<float>;
