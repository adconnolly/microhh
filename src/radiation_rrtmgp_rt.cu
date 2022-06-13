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

#include <numeric>
#include <boost/algorithm/string.hpp>

#include "radiation_rrtmgp_rt.h"
#include "radiation_rrtmgp_functions.h"
#include "grid.h"
#include "fields.h"
#include "timeloop.h"
#include "thermo.h"
#include "stats.h"
#include "netcdf_interface.h"
#include "constants.h"
#include "stats.h"
#include "cross.h"
#include "column.h"

#include "Array.h"
#include "Fluxes.h"
#include "Fluxes_rt.h"
#include "subset_kernel_launcher_cuda.h"
#include "rrtmgp_kernel_launcher_cuda_rt.h"
#include "gpt_combine_kernel_launcher_cuda_rt.h"

using namespace Radiation_rrtmgp_functions;

namespace
{
    __global__
    void calc_tendency_rt(
            Float* __restrict__ thlt_rad,
            const Float* __restrict__ rt_flux_abs_dir, const Float* __restrict__ rt_flux_abs_dif,
            const Float* __restrict__ rho, const Float* __restrict__ exner, const Float* __restrict__ dz,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            const int igc, const int jgc, const int kgc,
            const int jj, const int kk,
            const int jj_nogc, const int kk_nogc)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
        const int k = blockIdx.z + kstart;

        if ( (i < iend) && (j < jend) && (k < kend) )
        {
            const Float fac = Float(1.) / (rho[k] * Constants::cp<Float> * exner[k] * dz[k]);

            const int ijk = i + j*jj + k*kk;
            const int ijk_nogc = (i-igc) + (j-jgc)*jj_nogc + (k-kgc)*kk_nogc;

            thlt_rad[ijk] += fac * (rt_flux_abs_dir[ijk_nogc] + rt_flux_abs_dif[ijk_nogc]);
        }
    }

    __global__
    void calc_tendency(
            Float* __restrict__ thlt_rad,  const Float* __restrict__ flux_up,
            const Float* __restrict flux_dn, const Float* __restrict__ rho,
            const Float* __restrict__ exner, const Float* __restrict__ dz,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            const int igc, const int jgc, const int kgc,
            const int jj, const int kk,
            const int jj_nogc, const int kk_nogc)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
        const int k = blockIdx.z + kstart;

        if ( (i < iend) && (j < jend) && (k < kend) )
        {
            const Float fac = Float(1.) / (rho[k] * Constants::cp<Float> * exner[k] * dz[k]);

            const int ijk = i + j*jj + k*kk;
            const int ijk_nogc = (i-igc) + (j-jgc)*jj_nogc + (k-kgc)*kk_nogc;

            thlt_rad[ijk] -= fac * ( flux_up[ijk_nogc + kk_nogc] - flux_up[ijk_nogc]
                                   - flux_dn[ijk_nogc + kk_nogc] + flux_dn[ijk_nogc] );
        }
    }

    __global__
    void add_tendency(
            Float* __restrict__ thlt,  const Float* __restrict__ thlt_rad,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            const int jj, const int kk)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
        const int k = blockIdx.z + kstart;

        if ( (i < iend) && (j < jend) && (k < kend) )
        {
            const int ijk = i + j*jj + k*kk;
            thlt[ijk] += thlt_rad[ijk];
        }
    }

    __global__
    void store_surface_fluxes_rt(
            Float* __restrict__ flux_up_sfc, Float* __restrict__ flux_dn_sfc,
            const Float* __restrict__ rt_flux_sfc_dir, const Float* __restrict__ rt_flux_sfc_dif,
            const Float* __restrict__ rt_flux_sfc_up,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int igc, const int jgc,
            const int jj, const int kk,
            const int jj_nogc)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;

        if ( (i < iend) && (j < jend) )
        {
            const int ij = i + j*jj;
            const int ij_nogc = (i-igc) + (j-jgc)*jj_nogc;
            flux_up_sfc[ij] = rt_flux_sfc_up[ij_nogc];
            flux_dn_sfc[ij] = rt_flux_sfc_dir[ij_nogc] + rt_flux_sfc_dif[ij_nogc];
        }
    }

    __global__
    void store_surface_fluxes(
            Float* __restrict__ flux_up_sfc, Float* __restrict__ flux_dn_sfc,
            const Float* __restrict__ flux_up, const Float* __restrict__ flux_dn,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int igc, const int jgc,
            const int jj, const int kk,
            const int jj_nogc)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;

        if ( (i < iend) && (j < jend) )
        {
            const int ij = i + j*jj;
            const int ij_nogc = (i-igc) + (j-jgc)*jj_nogc;
            flux_up_sfc[ij] = flux_up[ij_nogc];
            flux_dn_sfc[ij] = flux_dn[ij_nogc];
        }
    }

    __global__
    void effective_radius_and_ciwp_to_gm2(
            Float* __restrict__ rel, Float* __restrict__ rei,
            Float* __restrict__ clwp, Float* __restrict__ ciwp,
            const Float* __restrict__ dz,
            const int ncol, const int nlay, const int kstart,
            const Float four_third_pi_N0_rho_w,
            const Float four_third_pi_N0_rho_i,
            const Float sig_g_fac)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;
            const int idx_z = ilay + kstart;
            const Float rel_local = clwp[idx] > Float(0.) ? Float(1.e6) * sig_g_fac * pow(clwp[idx] / dz[idx_z] / four_third_pi_N0_rho_w, Float(1.)/Float(3.)) : Float(0.);
            const Float rei_local = ciwp[idx] > Float(0.) ? Float(1.e6) * sig_g_fac * pow(ciwp[idx] / dz[idx_z] / four_third_pi_N0_rho_i, Float(1.)/Float(3.)) : Float(0.);

            rel[idx] = max(Float(2.5), min(rel_local, Float(21.5)));
            rei[idx] = max(Float(10.), min(rei_local, Float(180.)));

            clwp[idx] *= Float(1.e3);
            ciwp[idx] *= Float(1.e3);
        }
    }

    __global__
    void add_ghost_cells_g(
            Float* __restrict__ out, const Float* __restrict__ in,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kendh,
            const int jj, const int kk,
            const int jj_nogc, const int kk_nogc)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
        const int k = blockIdx.z + kstart;

        if (i < iend && j < jend && k < kendh)
        {
            const int ijk_nogc  = (i-istart) + (j-jstart)*jj_nogc + (k-kstart)*kk_nogc;
            const int ijk = i + j*jj + k*kk;

            out[ijk] = in[ijk_nogc];
        }
    }

    __global__
    void add_ghost_cells_2d_g(
            Float* __restrict__ out, const Float* __restrict__ in,
            const int istart, const int jstart,
            const int iend, const int jend,
            const int jj, const int jj_nogc)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;

        if (i < iend && j < jend)
        {
            const int ij_nogc  = (i-istart) + (j-jstart)*jj_nogc;
            const int ij = i + j*jj;

            out[ij] = in[ij_nogc];
        }
    }

    std::vector<std::string> get_variable_string(
            const std::string& var_name,
            std::vector<int> i_count,
            Netcdf_handle& input_nc,
            const int string_len,
            bool trim=true)
    {
        // Multiply all elements in i_count.
        int total_count = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Add the string length as the rightmost dimension.
        i_count.push_back(string_len);

        // Multiply all elements in i_count.
        // int total_count_char = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Read the entire char array;
        std::vector<char> var_char;
        var_char = input_nc.get_variable<char>(var_name, i_count);

        std::vector<std::string> var;

        for (int n=0; n<total_count; ++n)
        {
            std::string s(var_char.begin()+n*string_len, var_char.begin()+(n+1)*string_len);
            if (trim)
                boost::trim(s);
            var.push_back(s);
        }

        return var;
    }

    Gas_optics_rrtmgp_gpu load_and_init_gas_optics(
            Master& master,
            const Gas_concs_gpu& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        const int n_temps = coef_nc.get_dimension_size("temperature");
        const int n_press = coef_nc.get_dimension_size("pressure");
        const int n_absorbers = coef_nc.get_dimension_size("absorber");
        const int n_char = coef_nc.get_dimension_size("string_len");
        const int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        const int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        const int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        const int n_layers = coef_nc.get_dimension_size("atmos_layer");
        const int n_bnds = coef_nc.get_dimension_size("bnd");
        const int n_gpts = coef_nc.get_dimension_size("gpt");
        const int n_pairs = coef_nc.get_dimension_size("pair");
        const int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        const int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        const int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        const int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<Float,2> band_lims(coef_nc.get_variable<Float>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<Float,1> press_ref(coef_nc.get_variable<Float>("press_ref", {n_press}), {n_press});
        Array<Float,1> temp_ref(coef_nc.get_variable<Float>("temp_ref", {n_temps}), {n_temps});

        Float temp_ref_p = coef_nc.get_variable<Float>("absorption_coefficient_ref_P");
        Float temp_ref_t = coef_nc.get_variable<Float>("absorption_coefficient_ref_T");
        Float press_ref_trop = coef_nc.get_variable<Float>("press_ref_trop");

        Array<Float,3> kminor_lower(
                coef_nc.get_variable<Float>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<Float,3> kminor_upper(
                coef_nc.get_variable<Float>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
                                       {n_minorabsorbers});

        Array<std::string,1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_nc, n_char), {n_minorabsorbers});

        Array<std::string,1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,2> minor_limits_gpt_lower(
                coef_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int,2> minor_limits_gpt_upper(
                coef_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<Bool,1> minor_scales_with_density_lower(
                coef_nc.get_variable<Bool>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<Bool,1> minor_scales_with_density_upper(
                coef_nc.get_variable<Bool>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<Bool,1> scale_by_complement_lower(
                coef_nc.get_variable<Bool>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<Bool,1> scale_by_complement_upper(
                coef_nc.get_variable<Bool>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string,1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,1> kminor_start_lower(
                coef_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> kminor_start_upper(
                coef_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<Float,3> vmr_ref(
                coef_nc.get_variable<Float>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<Float,4> kmajor(
                coef_nc.get_variable<Float>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<Float,3> rayl_lower;
        Array<Float,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<Float>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<Float>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<Float,2> totplnk(
                    coef_nc.get_variable<Float>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<Float,4> planck_frac(
                    coef_nc.get_variable<Float>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics_rrtmgp_gpu(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    totplnk,
                    planck_frac,
                    rayl_lower,
                    rayl_upper);
        }
        else
        {
            Array<Float,1> solar_src_quiet(
                    coef_nc.get_variable<Float>("solar_source_quiet", {n_gpts}), {n_gpts});
            Array<Float,1> solar_src_facular(
                    coef_nc.get_variable<Float>("solar_source_facular", {n_gpts}), {n_gpts});
            Array<Float,1> solar_src_sunspot(
                    coef_nc.get_variable<Float>("solar_source_sunspot", {n_gpts}), {n_gpts});

            Float tsi = coef_nc.get_variable<Float>("tsi_default");
            Float mg_index = coef_nc.get_variable<Float>("mg_default");
            Float sb_index = coef_nc.get_variable<Float>("sb_default");

            return Gas_optics_rrtmgp_gpu(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    solar_src_quiet,
                    solar_src_facular,
                    solar_src_sunspot,
                    tsi,
                    mg_index,
                    sb_index,
                    rayl_lower,
                    rayl_upper);
        }
        // End reading of k-distribution.
    }

    Cloud_optics_gpu load_and_init_cloud_optics(
            Master& master,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("nband");
        int n_rghice   = coef_nc.get_dimension_size("nrghice");
        int n_size_liq = coef_nc.get_dimension_size("nsize_liq");
        int n_size_ice = coef_nc.get_dimension_size("nsize_ice");

        Array<Float,2> band_lims_wvn(coef_nc.get_variable<Float>("bnd_limits_wavenumber", {n_band, 2}), {2, n_band});

        // Read look-up table constants.
        Float radliq_lwr = coef_nc.get_variable<Float>("radliq_lwr");
        Float radliq_upr = coef_nc.get_variable<Float>("radliq_upr");
        Float radliq_fac = coef_nc.get_variable<Float>("radliq_fac");

        Float radice_lwr = coef_nc.get_variable<Float>("radice_lwr");
        Float radice_upr = coef_nc.get_variable<Float>("radice_upr");
        Float radice_fac = coef_nc.get_variable<Float>("radice_fac");

        Array<Float,2> lut_extliq(
                coef_nc.get_variable<Float>("lut_extliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<Float,2> lut_ssaliq(
                coef_nc.get_variable<Float>("lut_ssaliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<Float,2> lut_asyliq(
                coef_nc.get_variable<Float>("lut_asyliq", {n_band, n_size_liq}), {n_size_liq, n_band});

        Array<Float,3> lut_extice(
                coef_nc.get_variable<Float>("lut_extice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<Float,3> lut_ssaice(
                coef_nc.get_variable<Float>("lut_ssaice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<Float,3> lut_asyice(
                coef_nc.get_variable<Float>("lut_asyice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});

        return Cloud_optics_gpu(
                band_lims_wvn,
                radliq_lwr, radliq_upr, radliq_fac,
                radice_lwr, radice_upr, radice_fac,
                lut_extliq, lut_ssaliq, lut_asyliq,
                lut_extice, lut_ssaice, lut_asyice);
    }

    // extremely unnecessary copy of load_init_gas_optics - we definitely need a proper construction
    Gas_optics_rrtmgp_rt load_and_init_gas_optics_rt(
            Master& master,
            const Gas_concs_gpu& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        const int n_temps = coef_nc.get_dimension_size("temperature");
        const int n_press = coef_nc.get_dimension_size("pressure");
        const int n_absorbers = coef_nc.get_dimension_size("absorber");
        const int n_char = coef_nc.get_dimension_size("string_len");
        const int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        const int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        const int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        const int n_layers = coef_nc.get_dimension_size("atmos_layer");
        const int n_bnds = coef_nc.get_dimension_size("bnd");
        const int n_gpts = coef_nc.get_dimension_size("gpt");
        const int n_pairs = coef_nc.get_dimension_size("pair");
        const int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        const int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        const int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        const int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<Float,2> band_lims(coef_nc.get_variable<Float>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<Float,1> press_ref(coef_nc.get_variable<Float>("press_ref", {n_press}), {n_press});
        Array<Float,1> temp_ref(coef_nc.get_variable<Float>("temp_ref", {n_temps}), {n_temps});

        Float temp_ref_p = coef_nc.get_variable<Float>("absorption_coefficient_ref_P");
        Float temp_ref_t = coef_nc.get_variable<Float>("absorption_coefficient_ref_T");
        Float press_ref_trop = coef_nc.get_variable<Float>("press_ref_trop");

        Array<Float,3> kminor_lower(
                coef_nc.get_variable<Float>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<Float,3> kminor_upper(
                coef_nc.get_variable<Float>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
                                       {n_minorabsorbers});

        Array<std::string,1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_nc, n_char), {n_minorabsorbers});

        Array<std::string,1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,2> minor_limits_gpt_lower(
                coef_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int,2> minor_limits_gpt_upper(
                coef_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<Bool,1> minor_scales_with_density_lower(
                coef_nc.get_variable<Bool>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<Bool,1> minor_scales_with_density_upper(
                coef_nc.get_variable<Bool>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<Bool,1> scale_by_complement_lower(
                coef_nc.get_variable<Bool>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<Bool,1> scale_by_complement_upper(
                coef_nc.get_variable<Bool>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string,1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,1> kminor_start_lower(
                coef_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> kminor_start_upper(
                coef_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<Float,3> vmr_ref(
                coef_nc.get_variable<Float>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<Float,4> kmajor(
                coef_nc.get_variable<Float>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<Float,3> rayl_lower;
        Array<Float,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<Float>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<Float>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<Float,2> totplnk(
                    coef_nc.get_variable<Float>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<Float,4> planck_frac(
                    coef_nc.get_variable<Float>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics_rrtmgp_rt(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    totplnk,
                    planck_frac,
                    rayl_lower,
                    rayl_upper);
        }
        else
        {
            Array<Float,1> solar_src_quiet(
                    coef_nc.get_variable<Float>("solar_source_quiet", {n_gpts}), {n_gpts});
            Array<Float,1> solar_src_facular(
                    coef_nc.get_variable<Float>("solar_source_facular", {n_gpts}), {n_gpts});
            Array<Float,1> solar_src_sunspot(
                    coef_nc.get_variable<Float>("solar_source_sunspot", {n_gpts}), {n_gpts});

            Float tsi = coef_nc.get_variable<Float>("tsi_default");
            Float mg_index = coef_nc.get_variable<Float>("mg_default");
            Float sb_index = coef_nc.get_variable<Float>("sb_default");

            return Gas_optics_rrtmgp_rt(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    solar_src_quiet,
                    solar_src_facular,
                    solar_src_sunspot,
                    tsi,
                    mg_index,
                    sb_index,
                    rayl_lower,
                    rayl_upper);
        }
        // End reading of k-distribution.
    }

    Cloud_optics_rt load_and_init_cloud_optics_rt(
            Master& master,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("nband");
        int n_rghice   = coef_nc.get_dimension_size("nrghice");
        int n_size_liq = coef_nc.get_dimension_size("nsize_liq");
        int n_size_ice = coef_nc.get_dimension_size("nsize_ice");

        Array<Float,2> band_lims_wvn(coef_nc.get_variable<Float>("bnd_limits_wavenumber", {n_band, 2}), {2, n_band});

        // Read look-up table constants.
        Float radliq_lwr = coef_nc.get_variable<Float>("radliq_lwr");
        Float radliq_upr = coef_nc.get_variable<Float>("radliq_upr");
        Float radliq_fac = coef_nc.get_variable<Float>("radliq_fac");

        Float radice_lwr = coef_nc.get_variable<Float>("radice_lwr");
        Float radice_upr = coef_nc.get_variable<Float>("radice_upr");
        Float radice_fac = coef_nc.get_variable<Float>("radice_fac");

        Array<Float,2> lut_extliq(
                coef_nc.get_variable<Float>("lut_extliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<Float,2> lut_ssaliq(
                coef_nc.get_variable<Float>("lut_ssaliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<Float,2> lut_asyliq(
                coef_nc.get_variable<Float>("lut_asyliq", {n_band, n_size_liq}), {n_size_liq, n_band});

        Array<Float,3> lut_extice(
                coef_nc.get_variable<Float>("lut_extice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<Float,3> lut_ssaice(
                coef_nc.get_variable<Float>("lut_ssaice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<Float,3> lut_asyice(
                coef_nc.get_variable<Float>("lut_asyice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});

        return Cloud_optics_rt(
                band_lims_wvn,
                radliq_lwr, radliq_upr, radliq_fac,
                radice_lwr, radice_upr, radice_fac,
                lut_extliq, lut_ssaliq, lut_asyliq,
                lut_extice, lut_ssaice, lut_asyice);
    }

    void configure_memory_pool(int nlays, int ncols, int nchunks, int ngpts, int nbnds)
    {
        #ifdef RTE_RRTMGP_GPU_MEMPOOL_OWN
        /* Heuristic way to set up memory pool queues */
        std::map<std::size_t, std::size_t> pool_queues = {
            {64, 20},
            {128, 20},
            {256, 10},
            {512, 10},
            {1024, 5},
            {2048, 5},
            {nchunks * ngpts * sizeof(Float), 16},
            {nchunks * nbnds * sizeof(Float), 16},
            {(nlays + 1) * ncols * sizeof(Float), 14},
            {(nlays + 1) * nchunks * sizeof(Float), 10},
            {(nlays + 1) * nchunks * nbnds * sizeof(Float), 4},
            {(nlays + 1) * nchunks * ngpts * sizeof(int)/2, 6},
            {(nlays + 1) * nchunks * ngpts * sizeof(Float), 18}
        };

        Memory_pool_gpu::init_instance(pool_queues);
        #endif
    }
}


#ifdef USECUDA
template<typename TF>
void Radiation_rrtmgp_rt<TF>::prepare_device()
{
    auto& gd = grid.get_grid_data();

    // Set the memory pool.
    int ngpt_pool = 0;
    int nbnd_pool = 0;

    if (sw_longwave)
    {
        Netcdf_file coef_nc_lw(master, "coefficients_lw.nc", Netcdf_mode::Read);
        nbnd_pool = std::max(coef_nc_lw.get_dimension_size("bnd"), nbnd_pool);
        ngpt_pool = std::max(coef_nc_lw.get_dimension_size("gpt"), ngpt_pool);
    }

    if (sw_shortwave)
    {
        Netcdf_file coef_nc_sw(master, "coefficients_sw.nc", Netcdf_mode::Read);
        nbnd_pool = std::max(coef_nc_sw.get_dimension_size("bnd"), nbnd_pool);
        ngpt_pool = std::max(coef_nc_sw.get_dimension_size("gpt"), ngpt_pool);
    }

    configure_memory_pool(gd.ktot, gd.imax*gd.jmax, 512, ngpt_pool, nbnd_pool);


    // Initialize the pointers.
    this->gas_concs_gpu = std::make_unique<Gas_concs_gpu>(gas_concs);

    if (sw_longwave)
    {
        this->kdist_lw_gpu = std::make_unique<Gas_optics_rrtmgp_gpu>(
                load_and_init_gas_optics(master, *gas_concs_gpu, "coefficients_lw.nc"));

        this->cloud_lw_gpu = std::make_unique<Cloud_optics_gpu>(
                load_and_init_cloud_optics(master, "cloud_coefficients_lw.nc"));

        const int nsfcsize = gd.ijcells*sizeof(Float);
        cuda_safe_call(cudaMalloc(&lw_flux_dn_sfc_g, nsfcsize));
        cuda_safe_call(cudaMalloc(&lw_flux_up_sfc_g, nsfcsize));

        const int ncolgptsize = n_col*kdist_lw_gpu->get_ngpt()*sizeof(Float);
        cuda_safe_call(cudaMalloc(&lw_flux_dn_inc_g, ncolgptsize));

        cuda_safe_call(cudaMemcpy(lw_flux_dn_inc_g, lw_flux_dn_inc.ptr(), ncolgptsize, cudaMemcpyHostToDevice));
    }

    if (sw_shortwave)
    {
        this->kdist_sw_gpu = std::make_unique<Gas_optics_rrtmgp_gpu>(
                load_and_init_gas_optics(master, *gas_concs_gpu, "coefficients_sw.nc"));

        this->cloud_sw_gpu = std::make_unique<Cloud_optics_gpu>(
                load_and_init_cloud_optics(master, "cloud_coefficients_sw.nc"));

        this->kdist_sw_rt = std::make_unique<Gas_optics_rrtmgp_rt>(
                load_and_init_gas_optics_rt(master, *gas_concs_gpu, "coefficients_sw.nc"));

        this->cloud_sw_rt = std::make_unique<Cloud_optics_rt>(
                load_and_init_cloud_optics_rt(master, "cloud_coefficients_sw.nc"));

        const int nsfcsize = gd.ijcells*sizeof(Float);
        cuda_safe_call(cudaMalloc(&sw_flux_dn_sfc_g, nsfcsize));
        cuda_safe_call(cudaMalloc(&sw_flux_up_sfc_g, nsfcsize));

        cuda_safe_call(cudaMalloc(&sw_flux_sfc_dir_rt_g, nsfcsize));
        cuda_safe_call(cudaMalloc(&sw_flux_sfc_dif_rt_g, nsfcsize));
        cuda_safe_call(cudaMalloc(&sw_flux_sfc_up_rt_g, nsfcsize));
        cuda_safe_call(cudaMalloc(&sw_flux_tod_dn_rt_g, nsfcsize));
        cuda_safe_call(cudaMalloc(&sw_flux_tod_up_rt_g, nsfcsize));

        const int ncolgptsize = n_col*kdist_sw_rt->get_ngpt()*sizeof(Float);
        cuda_safe_call(cudaMalloc(&sw_flux_dn_dir_inc_g, ncolgptsize));
        cuda_safe_call(cudaMalloc(&sw_flux_dn_dif_inc_g, ncolgptsize));

        cuda_safe_call(cudaMemcpy(sw_flux_dn_dir_inc_g, sw_flux_dn_dir_inc.ptr(), ncolgptsize, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(sw_flux_dn_dif_inc_g, sw_flux_dn_dif_inc.ptr(), ncolgptsize, cudaMemcpyHostToDevice));
    }
}
#endif


#ifdef USECUDA
template<typename TF>
void Radiation_rrtmgp_rt<TF>::exec_longwave(
        Thermo<TF>& thermo, Timeloop<TF>& timeloop, Stats<TF>& stats,
        Array_gpu<Float,2>& flux_up, Array_gpu<Float,2>& flux_dn, Array_gpu<Float,2>& flux_net,
        const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev, const Array_gpu<Float,1>& t_sfc,
        const Array_gpu<Float,2>& h2o, const Array_gpu<Float,2>& clwp, const Array_gpu<Float,2>& ciwp,
        const bool compute_clouds)
{
    constexpr int n_col_block = 1024;

    auto& gd = grid.get_grid_data();

    const int n_col = gd.imax*gd.jmax;
    const int n_lay = gd.ktot;
    const int n_lev = gd.ktot+1;

    const int n_blocks = n_col / n_col_block;
    const int n_col_block_residual = n_col % n_col_block;

    const int n_gpt = this->kdist_lw_gpu->get_ngpt();
    const int n_bnd = this->kdist_lw_gpu->get_nband();

    const Bool top_at_1 = 0;

    // Define the pointers for the subsetting.
    std::unique_ptr<Optical_props_arry_gpu> optical_props_subset =
            std::make_unique<Optical_props_1scl_gpu>(n_col_block, n_lay, *kdist_lw_gpu);
    std::unique_ptr<Source_func_lw_gpu> sources_subset =
            std::make_unique<Source_func_lw_gpu>(n_col_block, n_lay, *kdist_lw_gpu);
    std::unique_ptr<Optical_props_1scl_gpu> cloud_optical_props_subset =
            std::make_unique<Optical_props_1scl_gpu>(n_col_block, n_lay, *cloud_lw_gpu);

    std::unique_ptr<Optical_props_arry_gpu> optical_props_residual =
            std::make_unique<Optical_props_1scl_gpu>(n_col_block_residual, n_lay, *kdist_lw_gpu);
    std::unique_ptr<Source_func_lw_gpu> sources_residual =
            std::make_unique<Source_func_lw_gpu>(n_col_block_residual, n_lay, *kdist_lw_gpu);
    std::unique_ptr<Optical_props_1scl_gpu> cloud_optical_props_residual =
            std::make_unique<Optical_props_1scl_gpu>(n_col_block_residual, n_lay, *cloud_lw_gpu);

    // Make view to the TOD flux pointers
    auto lw_flux_dn_inc_local = Array_gpu<Float,2>(lw_flux_dn_inc_g, {1, n_gpt});

    // Make views to the base state pointer.
    auto p_lay = Array_gpu<Float,2>(thermo.get_basestate_fld_g("pref") + gd.kstart, {1, n_lay});
    auto p_lev = Array_gpu<Float,2>(thermo.get_basestate_fld_g("prefh") + gd.kstart, {1, n_lev});

    // CvH: this can be improved by creating a fill function for the GPU.
    Array<Float,2> emis_sfc_cpu(std::vector<Float>(n_bnd, this->emis_sfc), {n_bnd, 1});
    Array_gpu<Float,2> emis_sfc(emis_sfc_cpu);

    gas_concs_gpu->set_vmr("h2o", h2o);

    // CvH: This can be done better: we now allocate a complete array.
    Array_gpu<Float,2> col_dry({n_col, n_lay});
    Gas_optics_rrtmgp_gpu::get_col_dry(col_dry, gas_concs_gpu->get_vmr("h2o"), p_lev.subset({{ {1, n_col}, {1, n_lev} }}));

    // Constants for computation of liquid and ice droplet effective radius
    const Float sig_g = 1.34;
    const Float fac = std::exp(std::log(sig_g)*std::log(sig_g)); // no conversion to micron yet.

    const Float Nc0 = 100.e6;
    const Float Ni0 = 1.e5;

    const Float four_third_pi_N0_rho_w = (4./3.)*M_PI*Nc0*Constants::rho_w<Float>;
    const Float four_third_pi_N0_rho_i = (4./3.)*M_PI*Ni0*Constants::rho_i<Float>;

    const int block_col = 16;
    const int block_lay = 16;
    const int grid_col  = n_col_block/block_col + (n_col_block%block_col > 0);
    const int grid_lay  = n_lay/block_lay + (n_lay%block_lay > 0);

    dim3 gridGPU_re (grid_col, grid_lay, 1);
    dim3 blockGPU_re (block_col, block_lay, 1);

    // Lambda function for solving optical properties subset.
    auto call_kernels = [&](
            const int col_s_in, const int col_e_in,
            std::unique_ptr<Optical_props_arry_gpu>& optical_props_subset_in,
            std::unique_ptr<Optical_props_1scl_gpu>& cloud_optical_props_subset_in,
            Source_func_lw_gpu& sources_subset_in,
            const Array_gpu<Float,2>& emis_sfc_subset_in,
            const Array_gpu<Float,2>& lw_flux_dn_inc_subset_in,
            Fluxes_broadband_gpu& fluxes,
            Fluxes_broadband_gpu& bnd_fluxes)
    {
        const int n_col_in = col_e_in - col_s_in + 1;
        Gas_concs_gpu gas_concs_subset(*gas_concs_gpu, col_s_in, n_col_in);

        auto p_lev_subset = p_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }});

        kdist_lw_gpu->gas_optics(
                p_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                p_lev_subset,
                t_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                t_sfc.subset({{ {col_s_in, col_e_in} }}),
                gas_concs_subset,
                optical_props_subset_in,
                sources_subset_in,
                col_dry.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                t_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }}) );


        if (compute_clouds)
        {
            auto clwp_subset = clwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }});
            auto ciwp_subset = ciwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }});
            Array_gpu<Float,2> rel({n_col_in, n_lay});
            Array_gpu<Float,2> rei({n_col_in, n_lay});

            effective_radius_and_ciwp_to_gm2<<<gridGPU_re, blockGPU_re>>>(
                    rel.ptr(), rei.ptr(),
                    clwp_subset.ptr(), ciwp_subset.ptr(),
                    gd.dz_g,
                    n_col_in, n_lay, gd.kstart,
                    four_third_pi_N0_rho_w, four_third_pi_N0_rho_i, fac);

            cloud_lw_gpu->cloud_optics(
                    clwp_subset,
                    ciwp_subset,
                    rel,
                    rei,
                    *cloud_optical_props_subset_in);

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_1scl_gpu&>(*optical_props_subset_in),
                    dynamic_cast<Optical_props_1scl_gpu&>(*cloud_optical_props_subset_in));
        }

        Array_gpu<Float,3> gpt_flux_up({n_col_in, n_lev, n_gpt});
        Array_gpu<Float,3> gpt_flux_dn({n_col_in, n_lev, n_gpt});

        constexpr int n_ang = 1;

        rte_lw_gpu.rte_lw(
                optical_props_subset_in,
                top_at_1,
                sources_subset_in,
                emis_sfc_subset_in,
                lw_flux_dn_inc_subset_in,
                gpt_flux_up,
                gpt_flux_dn,
                n_ang);

        fluxes.reduce(gpt_flux_up, gpt_flux_dn, optical_props_subset_in, top_at_1);

        // Copy the data to the output.
        subset_kernel_launcher_cuda::get_from_subset(
                n_col, n_lev, n_col_in, col_s_in, flux_up.ptr(), flux_dn.ptr(), flux_net.ptr(),
                fluxes.get_flux_up().ptr(), fluxes.get_flux_dn().ptr(), fluxes.get_flux_net().ptr());
    };

    for (int b=1; b<=n_blocks; ++b)
    {
        const int col_s = (b-1) * n_col_block + 1;
        const int col_e =  b    * n_col_block;

        Array_gpu<Float,2> emis_sfc_subset = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array_gpu<Float,2> lw_flux_dn_inc_subset = lw_flux_dn_inc_local.subset({{ {col_s, col_e}, {1, n_gpt} }});

        std::unique_ptr<Fluxes_broadband_gpu> fluxes_subset =
                std::make_unique<Fluxes_broadband_gpu>(n_col_block, n_lev);
        std::unique_ptr<Fluxes_broadband_gpu> bnd_fluxes_subset =
                std::make_unique<Fluxes_byband_gpu>(n_col_block, n_lev, n_bnd);

        call_kernels(
                col_s, col_e,
                optical_props_subset,
                cloud_optical_props_subset,
                *sources_subset,
                emis_sfc_subset,
                lw_flux_dn_inc_subset,
                *fluxes_subset,
                *bnd_fluxes_subset);
    }

    if (n_col_block_residual > 0)
    {
        const int col_s = n_col - n_col_block_residual + 1;
        const int col_e = n_col;

        Array_gpu<Float,2> emis_sfc_residual = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array_gpu<Float,2> lw_flux_dn_inc_residual = lw_flux_dn_inc_local.subset({{ {col_s, col_e}, {1, n_gpt} }});

        std::unique_ptr<Fluxes_broadband_gpu> fluxes_residual =
                std::make_unique<Fluxes_broadband_gpu>(n_col_block_residual, n_lev);
        std::unique_ptr<Fluxes_broadband_gpu> bnd_fluxes_residual =
                std::make_unique<Fluxes_byband_gpu>(n_col_block_residual, n_lev, n_bnd);

        call_kernels(
                col_s, col_e,
                optical_props_residual,
                cloud_optical_props_residual,
                *sources_residual,
                emis_sfc_residual,
                lw_flux_dn_inc_residual,
                *fluxes_residual,
                *bnd_fluxes_residual);
    }
}
#endif

#ifdef USECUDA
template<typename TF>
void Radiation_rrtmgp_rt<TF>::exec_shortwave(
        Thermo<TF>& thermo, Timeloop<TF>& timeloop, Stats<TF>& stats,
        Array_gpu<Float,2>& flux_up, Array_gpu<Float,2>& flux_dn, Array_gpu<Float,2>& flux_dn_dir, Array_gpu<Float,2>& flux_net,
        const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
        const Array_gpu<Float,2>& h2o, const Array_gpu<Float,2>& clwp, const Array_gpu<Float,2>& ciwp,
        const bool compute_clouds)
{
    constexpr int n_col_block = 1024;

    auto& gd = grid.get_grid_data();

    const int n_col = gd.imax*gd.jmax;
    const int n_lay = gd.ktot;
    const int n_lev = gd.ktot+1;

    const int n_blocks = n_col / n_col_block;
    const int n_col_block_residual = n_col % n_col_block;

    const int n_gpt = this->kdist_sw_gpu->get_ngpt();
    const int n_bnd = this->kdist_sw_gpu->get_nband();

    const Bool top_at_1 = 0;

    // Define the pointers for the subsetting.
    std::unique_ptr<Optical_props_arry_gpu> optical_props_subset =
            std::make_unique<Optical_props_2str_gpu>(n_col_block, n_lay, *kdist_sw_gpu);
    std::unique_ptr<Optical_props_2str_gpu> cloud_optical_props_subset =
            std::make_unique<Optical_props_2str_gpu>(n_col_block, n_lay, *cloud_sw_gpu);

    std::unique_ptr<Optical_props_arry_gpu> optical_props_residual =
            std::make_unique<Optical_props_2str_gpu>(n_col_block_residual, n_lay, *kdist_sw_gpu);
    std::unique_ptr<Optical_props_2str_gpu> cloud_optical_props_residual =
            std::make_unique<Optical_props_2str_gpu>(n_col_block_residual, n_lay, *cloud_sw_gpu);

    // Make views to the base state pointer.
    auto p_lay = Array_gpu<Float,2>(thermo.get_basestate_fld_g("pref") + gd.kstart, {1, n_lay});
    auto p_lev = Array_gpu<Float,2>(thermo.get_basestate_fld_g("prefh") + gd.kstart, {1, n_lev});

    // Make views to the TOD flux pointers
    auto sw_flux_dn_dir_inc_local = Array_gpu<Float,2>(sw_flux_dn_dir_inc_g, {1, n_gpt});
    auto sw_flux_dn_dif_inc_local = Array_gpu<Float,2>(sw_flux_dn_dif_inc_g, {1, n_gpt});

    // Create the boundary conditions
    Array<Float,1> mu0_cpu(std::vector<Float>(1, this->mu0), {1});
    Array_gpu<Float,1> mu0(mu0_cpu);
    Array<Float,2> sfc_alb_dir_cpu(std::vector<Float>(n_bnd, this->sfc_alb_dir), {n_bnd, 1});
    Array_gpu<Float,2> sfc_alb_dir(sfc_alb_dir_cpu);
    Array<Float,2> sfc_alb_dif_cpu(std::vector<Float>(n_bnd, this->sfc_alb_dif), {n_bnd, 1});
    Array_gpu<Float,2> sfc_alb_dif(sfc_alb_dif_cpu);

    gas_concs_gpu->set_vmr("h2o", h2o);

    // CvH: This can be done better: we now allocate a complete array.
    Array_gpu<Float,2> col_dry({n_col, n_lay});
    Gas_optics_rrtmgp_gpu::get_col_dry(col_dry, gas_concs_gpu->get_vmr("h2o"), p_lev.subset({{ {1, n_col}, {1, n_lev} }}));

    // Constants for computation of liquid and ice droplet effective radius
    const Float sig_g = 1.34;
    const Float fac = std::exp(std::log(sig_g)*std::log(sig_g)); // no conversion to micron yet.

    const Float Nc0 = 100.e6;
    const Float Ni0 = 1.e5;

    const Float four_third_pi_N0_rho_w = (4./3.)*M_PI*Nc0*Constants::rho_w<Float>;
    const Float four_third_pi_N0_rho_i = (4./3.)*M_PI*Ni0*Constants::rho_i<Float>;

    const int block_col = 16;
    const int block_lay = 16;
    const int grid_col  = n_col_block/block_col + (n_col_block%block_col > 0);
    const int grid_lay  = n_lay/block_lay + (n_lay%block_lay > 0);

    dim3 gridGPU_re (grid_col, grid_lay, 1);
    dim3 blockGPU_re (block_col, block_lay, 1);

    // Lambda function for solving optical properties subset.
    auto call_kernels = [&](
            const int col_s_in, const int col_e_in,
            std::unique_ptr<Optical_props_arry_gpu>& optical_props_subset_in,
            std::unique_ptr<Optical_props_2str_gpu>& cloud_optical_props_subset_in,
            const Array_gpu<Float,1>& mu0_subset_in,
            const Array_gpu<Float,2>& sw_flux_dn_dir_inc_subset_in,
            const Array_gpu<Float,2>& sfc_alb_dir_subset_in,
            const Array_gpu<Float,2>& sfc_alb_dif_subset_in,
            const Array_gpu<Float,2>& sw_flux_dn_dif_inc_subset_in,
            Fluxes_broadband_gpu& fluxes,
            Fluxes_broadband_gpu& bnd_fluxes)
    {
        const int n_col_in = col_e_in - col_s_in + 1;
        Gas_concs_gpu gas_concs_subset(*gas_concs_gpu, col_s_in, n_col_in);
        Array_gpu<Float,2> toa_src_dummy({n_col_in, n_gpt});

        auto p_lev_subset = p_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }});
        kdist_sw_gpu->gas_optics(
                p_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                p_lev_subset,
                t_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                gas_concs_subset,
                optical_props_subset_in,
                toa_src_dummy,
                col_dry.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}) );


        if (compute_clouds)
        {
            auto clwp_subset = clwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }});
            auto ciwp_subset = ciwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }});
            Array_gpu<Float,2> rel({n_col_in, n_lay});
            Array_gpu<Float,2> rei({n_col_in, n_lay});

            effective_radius_and_ciwp_to_gm2<<<gridGPU_re, blockGPU_re>>>(
                    rel.ptr(), rei.ptr(),
                    clwp_subset.ptr(), ciwp_subset.ptr(),
                    gd.dz_g,
                    n_col_in, n_lay, gd.kstart,
                    four_third_pi_N0_rho_w, four_third_pi_N0_rho_i, fac);

            cloud_sw_gpu->cloud_optics(
                    clwp_subset,
                    ciwp_subset,
                    rel,
                    rei,
                    *cloud_optical_props_subset_in);

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_2str_gpu&>(*optical_props_subset_in),
                    dynamic_cast<Optical_props_2str_gpu&>(*cloud_optical_props_subset_in));
        }

        Array_gpu<Float,3> gpt_flux_up({n_col_in, n_lev, n_gpt});
        Array_gpu<Float,3> gpt_flux_dn({n_col_in, n_lev, n_gpt});
        Array_gpu<Float,3> gpt_flux_dn_dir({n_col_in, n_lev, n_gpt});

        rte_sw_gpu.rte_sw(
                optical_props_subset_in,
                top_at_1,
                mu0_subset_in,
                sw_flux_dn_dir_inc_subset_in,
                sfc_alb_dir_subset_in,
                sfc_alb_dif_subset_in,
                sw_flux_dn_dif_inc_subset_in,
                gpt_flux_up,
                gpt_flux_dn,
                gpt_flux_dn_dir);

        fluxes.reduce(gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir, optical_props_subset_in, top_at_1);

        // Copy the data to the output.
        subset_kernel_launcher_cuda::get_from_subset(
                n_col, n_lev, n_col_in, col_s_in, flux_up.ptr(), flux_dn.ptr(), flux_dn_dir.ptr(), flux_net.ptr(),
                fluxes.get_flux_up().ptr(), fluxes.get_flux_dn().ptr(), fluxes.get_flux_dn_dir().ptr(), fluxes.get_flux_net().ptr());
    };

    for (int b=1; b<=n_blocks; ++b)
    {
        const int col_s = (b-1) * n_col_block + 1;
        const int col_e =  b    * n_col_block;

        Array_gpu<Float,1> mu0_subset = mu0.subset({{ {col_s, col_e} }});
        Array_gpu<Float,2> sw_flux_dn_dir_inc_subset = sw_flux_dn_dir_inc_local.subset({{ {col_s, col_e}, {1, n_gpt} }});
        Array_gpu<Float,2> sfc_alb_dir_subset = sfc_alb_dir.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array_gpu<Float,2> sfc_alb_dif_subset = sfc_alb_dif.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array_gpu<Float,2> sw_flux_dn_dif_inc_subset = sw_flux_dn_dif_inc_local.subset({{ {col_s, col_e}, {1, n_gpt} }});

        std::unique_ptr<Fluxes_broadband_gpu> fluxes_subset =
                std::make_unique<Fluxes_broadband_gpu>(n_col_block, n_lev);
        std::unique_ptr<Fluxes_broadband_gpu> bnd_fluxes_subset =
                std::make_unique<Fluxes_byband_gpu>(n_col_block, n_lev, n_bnd);
        call_kernels(
                col_s, col_e,
                optical_props_subset,
                cloud_optical_props_subset,
                mu0_subset,
                sw_flux_dn_dir_inc_subset,
                sfc_alb_dir_subset,
                sfc_alb_dif_subset,
                sw_flux_dn_dif_inc_subset,
                *fluxes_subset,
                *bnd_fluxes_subset);

    }

    if (n_col_block_residual > 0)
    {
        const int col_s = n_col - n_col_block_residual + 1;
        const int col_e = n_col;

        Array_gpu<Float,1> mu0_residual = mu0.subset({{ {col_s, col_e} }});
        Array_gpu<Float,2> sw_flux_dn_dir_inc_residual = sw_flux_dn_dir_inc_local.subset({{ {col_s, col_e}, {1, n_gpt} }});
        Array_gpu<Float,2> sfc_alb_dir_residual = sfc_alb_dir.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array_gpu<Float,2> sfc_alb_dif_residual = sfc_alb_dif.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array_gpu<Float,2> sw_flux_dn_dif_inc_residual = sw_flux_dn_dif_inc_local.subset({{ {col_s, col_e}, {1, n_gpt} }});

        std::unique_ptr<Fluxes_broadband_gpu> fluxes_residual =
                std::make_unique<Fluxes_broadband_gpu>(n_col_block_residual, n_lev);
        std::unique_ptr<Fluxes_broadband_gpu> bnd_fluxes_residual =
                std::make_unique<Fluxes_byband_gpu>(n_col_block_residual, n_lev, n_bnd);

        call_kernels(
                col_s, col_e,
                optical_props_residual,
                cloud_optical_props_residual,
                mu0_residual,
                sw_flux_dn_dir_inc_residual,
                sfc_alb_dir_residual,
                sfc_alb_dif_residual,
                sw_flux_dn_dif_inc_residual,
                *fluxes_residual,
                *bnd_fluxes_residual);
    }
}
#endif

#ifdef USECUDA
template<typename TF>
void Radiation_rrtmgp_rt<TF>::exec_shortwave_rt(
        Thermo<TF>& thermo, Timeloop<TF>& timeloop, Stats<TF>& stats,
        Array_gpu<Float,2>& flux_up, Array_gpu<Float,2>& flux_dn, Array_gpu<Float,2>& flux_dn_dir, Array_gpu<Float,2>& flux_net,
        Array_gpu<Float,2>& rt_flux_tod_dn, Array_gpu<Float,2>& rt_flux_tod_up, Array_gpu<Float,2>& rt_flux_sfc_dir, Array_gpu<Float,2>& rt_flux_sfc_dif,
        Array_gpu<Float,2>& rt_flux_sfc_up, Array_gpu<Float,3>& rt_flux_abs_dir, Array_gpu<Float,3>& rt_flux_abs_dif,
        const Array_gpu<Float,2>& t_lay, const Array_gpu<Float,2>& t_lev,
        const Array_gpu<Float,2>& h2o, Array_gpu<Float,2>& clwp, Array_gpu<Float,2>& ciwp,
        const bool compute_clouds)
{
    auto& gd = grid.get_grid_data();

    const int n_col = gd.imax*gd.jmax;
    const int n_lay = gd.ktot;
    const int n_lev = gd.ktot+1;

    const int n_gpt = this->kdist_sw_rt->get_ngpt();
    const int n_bnd = this->kdist_sw_rt->get_nband();

    const Bool top_at_1 = 0;

    // initiate flux & heating rate arrays to 0
    rrtmgp_kernel_launcher_cuda_rt::zero_array(gd.jmax, gd.imax, rt_flux_tod_dn.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(gd.jmax, gd.imax, rt_flux_tod_up.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(gd.jmax, gd.imax, rt_flux_sfc_dir.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(gd.jmax, gd.imax, rt_flux_sfc_dif.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(gd.jmax, gd.imax, rt_flux_sfc_up.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(n_lay, gd.jmax, gd.imax, rt_flux_abs_dir.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(n_lay, gd.jmax, gd.imax, rt_flux_abs_dif.ptr());

    rrtmgp_kernel_launcher_cuda_rt::zero_array(n_lev, gd.jmax, gd.imax, flux_up.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(n_lev, gd.jmax, gd.imax, flux_dn.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(n_lev, gd.jmax, gd.imax, flux_dn_dir.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(n_lev, gd.jmax, gd.imax, flux_net.ptr());


    // Define the pointers for the subsetting.
    std::unique_ptr<Optical_props_arry_rt> optical_props =
            std::make_unique<Optical_props_2str_rt>(n_col, n_lay, *kdist_sw_rt);
    std::unique_ptr<Optical_props_2str_rt> cloud_optical_props =
            std::make_unique<Optical_props_2str_rt>(n_col, n_lay, *cloud_sw_rt);

    // Make views to the base state pointer.
    auto p_lay_tmp = Array_gpu<Float,2>(thermo.get_basestate_fld_g("pref") + gd.kstart, {1, n_lay});
    auto p_lev_tmp = Array_gpu<Float,2>(thermo.get_basestate_fld_g("prefh") + gd.kstart, {1, n_lev});

    // // Make TOD flux arrays
    Array_gpu<Float,1> sw_flux_dn_dir_inc_local({n_col});// = Array_gpu<Float,2>(sw_flux_dn_dir_inc_g, {1, n_gpt});
    Array_gpu<Float,1> sw_flux_dn_dif_inc_local({n_col});// = Array_gpu<Float,2>(sw_flux_dn_dif_inc_g, {1, n_gpt});

    // Create the boundary conditions
    Array<Float,1> mu0_cpu(std::vector<Float>(1, this->mu0), {1});
    Array_gpu<Float,1> mu0(mu0_cpu);
    Array<Float,2> sfc_alb_dir_cpu(std::vector<Float>(n_bnd, this->sfc_alb_dir), {n_bnd, 1});
    Array_gpu<Float,2> sfc_alb_dir(sfc_alb_dir_cpu);
    Array<Float,2> sfc_alb_dif_cpu(std::vector<Float>(n_bnd, this->sfc_alb_dif), {n_bnd, 1});
    Array_gpu<Float,2> sfc_alb_dif(sfc_alb_dif_cpu);

    gas_concs_gpu->set_vmr("h2o", h2o);

    // plev and play need column dimension
    auto p_lay = p_lay_tmp.subset({{ {1, n_col}, {1, n_lay} }});
    auto p_lev = p_lev_tmp.subset({{ {1, n_col}, {1, n_lev} }});

    // CvH: This can be done better: we now allocate a complete array.
    Array_gpu<Float,2> col_dry({n_col, n_lay});
    Gas_optics_rrtmgp_rt::get_col_dry(col_dry, gas_concs_gpu->get_vmr("h2o"), p_lev);

    Array_gpu<Float,1> toa_src_dummy({n_col});

    // compute cloud eff radius and convert ice/liquid water path to g/m2
    Array_gpu<Float,2> rel;
    Array_gpu<Float,2> rei;
    if (compute_clouds)
    {
        // Constants for computation of liquid and ice droplet effective radius
        const Float sig_g = 1.34;
        const Float fac = std::exp(std::log(sig_g)*std::log(sig_g)); // no conversion to micron yet.

        const Float Nc0 = 100.e6;
        const Float Ni0 = 1.e5;

        const Float four_third_pi_N0_rho_w = (4./3.)*M_PI*Nc0*Constants::rho_w<Float>;
        const Float four_third_pi_N0_rho_i = (4./3.)*M_PI*Ni0*Constants::rho_i<Float>;

        const int block_col = 16;
        const int block_lay = 16;
        const int grid_col  = n_col/block_col + (n_col%block_col > 0);
        const int grid_lay  = n_lay/block_lay + (n_lay%block_lay > 0);

        dim3 gridGPU_re (grid_col, grid_lay, 1);
        dim3 blockGPU_re (block_col, block_lay, 1);

        rel.set_dims({n_col, n_lay});
        rei.set_dims({n_col, n_lay});

        effective_radius_and_ciwp_to_gm2<<<gridGPU_re, blockGPU_re>>>(
                rel.ptr(), rei.ptr(),
                clwp.ptr(), ciwp.ptr(),
                gd.dz_g,
                n_col, n_lay, gd.kstart,
                four_third_pi_N0_rho_w, four_third_pi_N0_rho_i, fac);
    }

    // main g-point loop
    const Array<int, 2>& band_limits_gpt(this->kdist_sw_rt->get_band_lims_gpoint());
    for (int igpt=1; igpt<=n_gpt; ++igpt)
    {
        int band = 0;
        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
        {
            if (igpt <= band_limits_gpt({2, ibnd}))
            {
                band = ibnd;
                break;
            }
        }

        kdist_sw_rt->gas_optics(
                igpt-1,
                p_lay,
                p_lev,
                t_lay,
                *gas_concs_gpu,
                optical_props,
                toa_src_dummy,
                col_dry);

        if (compute_clouds)
        {
            cloud_sw_rt->cloud_optics(
                    band-1,
                    clwp,
                    ciwp,
                    rel,
                    rei,
                    *cloud_optical_props);

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_2str_rt&>(*optical_props),
                    dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props));
        }

        std::unique_ptr<Fluxes_broadband_rt> fluxes =
                std::make_unique<Fluxes_broadband_rt>(gd.imax, gd.jmax, n_lev);

        sw_flux_dn_dir_inc_local.fill(sw_flux_dn_dir_inc({1, igpt}));
        sw_flux_dn_dif_inc_local.fill(sw_flux_dn_dif_inc({1, igpt}));

        rte_sw_rt.rte_sw(
                optical_props,
                top_at_1,
                mu0.subset({{ {1, n_col} }}),
                sw_flux_dn_dir_inc_local,
                sfc_alb_dir.subset({{ {band, band}, {1, n_col}} }),
                sfc_alb_dif.subset({{ {band, band}, {1, n_col}} }),
                sw_flux_dn_dif_inc_local,
                (*fluxes).get_flux_up(),
                (*fluxes).get_flux_dn(),
                (*fluxes).get_flux_dn_dir());

        // CvH: this computation assumes that mu0 and azimuth are constant over the entire subset. Works for small LES only.
        Float zenith_angle = std::acos(mu0({1}));
        Float azimuth_angle = this->azimuth;

        const int ngrid_x = 52;
        const int ngrid_y = 52;
        const int ngrid_z = 16;
        const Int photons_per_pixel = Int(256);
        raytracer.trace_rays(
                photons_per_pixel,
                gd.imax, gd.jmax, n_lay,
                gd.dx, gd.dy, gd.dz[gd.kstart],
                ngrid_x, ngrid_y, ngrid_z,
                dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_tau(),
                dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_ssa(),
                dynamic_cast<Optical_props_2str_rt&>(*optical_props).get_g(),
                dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props).get_tau(),
                sfc_alb_dir, zenith_angle,
                azimuth_angle,
                sw_flux_dn_dir_inc({1,igpt}) * mu0({1}), sw_flux_dn_dif_inc({1,igpt}),
                (*fluxes).get_flux_tod_dn(),
                (*fluxes).get_flux_tod_up(),
                (*fluxes).get_flux_sfc_dir(),
                (*fluxes).get_flux_sfc_dif(),
                (*fluxes).get_flux_sfc_up(),
                (*fluxes).get_flux_abs_dir(),
                (*fluxes).get_flux_abs_dif());

        (*fluxes).net_flux();

        gpt_combine_kernel_launcher_cuda_rt::add_from_gpoint(
                  n_col, n_lev, flux_up.ptr(), flux_dn.ptr(), flux_dn_dir.ptr(), flux_net.ptr(),
                  (*fluxes).get_flux_up().ptr(), (*fluxes).get_flux_dn().ptr(), (*fluxes).get_flux_dn_dir().ptr(), (*fluxes).get_flux_net().ptr());

        gpt_combine_kernel_launcher_cuda_rt::add_from_gpoint(
                  gd.imax, gd.jmax, rt_flux_tod_dn.ptr(), rt_flux_tod_up.ptr(), rt_flux_sfc_dir.ptr(), rt_flux_sfc_dif.ptr(), rt_flux_sfc_up.ptr(),
                  (*fluxes).get_flux_tod_dn().ptr(), (*fluxes).get_flux_tod_up().ptr(), (*fluxes).get_flux_sfc_dir().ptr(), (*fluxes).get_flux_sfc_dif().ptr(), (*fluxes).get_flux_sfc_up().ptr());

        gpt_combine_kernel_launcher_cuda_rt::add_from_gpoint(
                  n_col, n_lay, rt_flux_abs_dir.ptr(), rt_flux_abs_dif.ptr(),
                  (*fluxes).get_flux_abs_dir().ptr(), (*fluxes).get_flux_abs_dif().ptr());

    }
}
#endif

#ifdef USECUDA
template <typename TF>
void Radiation_rrtmgp_rt<TF>::exec(Thermo<TF>& thermo, double time, Timeloop<TF>& timeloop, Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;
    const int gridi  = gd.imax/blocki + (gd.imax%blocki > 0);
    const int gridj  = gd.jmax/blockj + (gd.jmax%blockj > 0);

    dim3 gridGPU_3d (gridi, gridj, gd.kmax+1);
    dim3 blockGPU_3d(blocki, blockj, 1);
    dim3 gridGPU_2d (gridi, gridj, 1);
    dim3 blockGPU_2d(blocki, blockj, 1);

    const bool do_radiation = ((timeloop.get_itime() % idt_rad == 0) && !timeloop.in_substep()) ;
    const bool do_radiation_stats = timeloop.is_stats_step();

    if (do_radiation)
    {
        // Set the tendency to zero.
        // std::fill(fields.sd.at("thlt_rad")->fld.begin(), fields.sd.at("thlt_rad")->fld.end(), Float(0.));
        cudaMemset(fields.sd.at("thlt_rad")->fld_g, 0, gd.ncells*sizeof(Float));

        auto t_lay = fields.get_tmp_g();
        auto t_lev = fields.get_tmp_g();
        auto h2o   = fields.get_tmp_g(); // This is the volume mixing ratio, not the specific humidity of vapor.
        auto clwp  = fields.get_tmp_g();
        auto ciwp  = fields.get_tmp_g();

        // Set the input to the radiation on a 3D grid without ghost cells.
        thermo.get_radiation_fields_g(*t_lay, *t_lev, *h2o, *clwp, *ciwp);

        const int nmaxh = gd.imax*gd.jmax*(gd.ktot+1);
        const int ijmax = gd.imax*gd.jmax;

        // Create views on existing variables.
        Array_gpu<Float,2> t_lay_a(t_lay->fld_g, {gd.imax*gd.jmax, gd.ktot});
        Array_gpu<Float,2> t_lev_a(t_lev->fld_g, {gd.imax*gd.jmax, gd.ktot+1});
        Array_gpu<Float,1> t_sfc_a(t_lev->fld_bot_g, {gd.imax*gd.jmax});
        Array_gpu<Float,2> h2o_a(h2o->fld_g, {gd.imax*gd.jmax, gd.ktot});
        Array_gpu<Float,2> clwp_a(clwp->fld_g, {gd.imax*gd.jmax, gd.ktot});
        Array_gpu<Float,2> ciwp_a(ciwp->fld_g, {gd.imax*gd.jmax, gd.ktot});

        const bool compute_clouds = true;

        try
        {
            if (sw_longwave)
            {
                // Flux fields.
                Array_gpu<Float,2> flux_up ({gd.imax*gd.jmax, gd.ktot+1});
                Array_gpu<Float,2> flux_dn ({gd.imax*gd.jmax, gd.ktot+1});
                Array_gpu<Float,2> flux_net({gd.imax*gd.jmax, gd.ktot+1});

                exec_longwave(
                        thermo, timeloop, stats,
                        flux_up, flux_dn, flux_net,
                        t_lay_a, t_lev_a, t_sfc_a, h2o_a, clwp_a, ciwp_a,
                        compute_clouds);

                calc_tendency<<<gridGPU_3d, blockGPU_3d>>>(
                        fields.sd.at("thlt_rad")->fld_g,
                        flux_up.ptr(), flux_dn.ptr(),
                        fields.rhoref_g, thermo.get_basestate_fld_g("exner"),
                        gd.dz_g,
                        gd.istart, gd.jstart, gd.kstart,
                        gd.iend, gd.jend, gd.kend,
                        gd.igc, gd.jgc, gd.kgc,
                        gd.icells, gd.ijcells,
                        gd.imax, gd.imax*gd.jmax);
                cuda_check_error();

                store_surface_fluxes<<<gridGPU_2d, blockGPU_2d>>>(
                        lw_flux_up_sfc_g, lw_flux_dn_sfc_g,
                        flux_up.ptr(), flux_dn.ptr(),
                        gd.istart, gd.iend,
                        gd.jstart, gd.jend,
                        gd.igc, gd.jgc,
                        gd.icells, gd.ijcells,
                        gd.imax);
                cuda_check_error();

                if (do_radiation_stats)
                {
                    // Make sure that the top boundary is taken into account in case of fluxes.
                    auto do_gcs = [&](Field3d<Float>& out, const Array_gpu<Float,2>& in)
                    {
                        add_ghost_cells_g<<<gridGPU_3d, blockGPU_3d>>>(
                                out.fld_g, in.ptr(),
                                gd.istart, gd.jstart, gd.kstart,
                                gd.iend, gd.jend, gd.kend+1,
                                gd.icells, gd.ijcells,
                                gd.imax, gd.imax*gd.jmax);
                    };

                    do_gcs(*fields.sd.at("lw_flux_up"), flux_up);
                    do_gcs(*fields.sd.at("lw_flux_dn"), flux_dn);

                    // clear sky
                    if (sw_clear_sky_stats)
                    {
                        exec_longwave(
                                thermo, timeloop, stats,
                                flux_up, flux_dn, flux_net,
                                t_lay_a, t_lev_a, t_sfc_a, h2o_a, clwp_a, ciwp_a,
                                !compute_clouds);

                        do_gcs(*fields.sd.at("lw_flux_up_clear"), flux_up);
                        do_gcs(*fields.sd.at("lw_flux_dn_clear"), flux_dn);
                    }
                }
            }

            if (sw_shortwave)
            {
                // Flux fields.
                Array_gpu<Float,2> flux_up ({gd.imax*gd.jmax, gd.ktot+1});
                Array_gpu<Float,2> flux_dn ({gd.imax*gd.jmax, gd.ktot+1});
                Array_gpu<Float,2> flux_net({gd.imax*gd.jmax, gd.ktot+1});
                Array_gpu<Float,2> flux_dn_dir({gd.imax*gd.jmax, gd.ktot+1});

                // RT fields.
                Array_gpu<Float,2> rt_flux_tod_dn({gd.imax,gd.jmax});
                Array_gpu<Float,2> rt_flux_tod_up({gd.imax,gd.jmax});
                Array_gpu<Float,2> rt_flux_sfc_dir({gd.imax,gd.jmax});
                Array_gpu<Float,2> rt_flux_sfc_dif({gd.imax,gd.jmax});
                Array_gpu<Float,2> rt_flux_sfc_up({gd.imax,gd.jmax});
                Array_gpu<Float,3> rt_flux_abs_dir({gd.imax,gd.jmax, gd.ktot});
                Array_gpu<Float,3> rt_flux_abs_dif({gd.imax,gd.jmax, gd.ktot});

                // Single column solve of background profile for TOA conditions
                if (!sw_fixed_sza)
                {
                    // Update the solar zenith angle, and calculate new shortwave reference column
                    const int day_of_year = int(timeloop.calc_day_of_year());
                    const int year = timeloop.get_year();
                    const Float seconds_after_midnight = Float(timeloop.calc_hour_of_day()*3600);
                    std::tie(this->mu0, this->azimuth) = calc_cos_zenith_angle(lat, lon, day_of_year, seconds_after_midnight, year);

                    // Calculate correction factor for impact Sun's distance on the solar "constant"
                    const Float frac_day_of_year = Float(day_of_year) + seconds_after_midnight / Float(86400);
                    this->tsi_scaling = calc_sun_distance_factor(frac_day_of_year);

                    if (is_day(this->mu0))
                    {
                        const int n_bnd = kdist_sw->get_nband();
                        const int n_gpt = kdist_sw->get_ngpt();

                        // Set the solar zenith angle and albedo.
                        Array<Float,2> sfc_alb_dir({n_bnd, n_col});
                        Array<Float,2> sfc_alb_dif({n_bnd, n_col});

                        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
                        {
                            sfc_alb_dir({ibnd, 1}) = this->sfc_alb_dir;
                            sfc_alb_dif({ibnd, 1}) = this->sfc_alb_dif;
                        }

                        Array<Float,1> mu0({n_col});
                        mu0({1}) = this->mu0;

                        // sw column solve on cpu for TOD fluxes
                        solve_shortwave_column(
                                optical_props_sw,
                                sw_flux_up_col, sw_flux_dn_col, sw_flux_dn_dir_col, sw_flux_net_col,
                                sw_flux_dn_dir_inc, sw_flux_dn_dif_inc, thermo.get_basestate_vector("ph")[gd.kend],
                                gas_concs_col,
                                *kdist_sw,
                                col_dry,
                                p_lay_col, p_lev_col,
                                t_lay_col, t_lev_col,
                                mu0,
                                sfc_alb_dir, sfc_alb_dif,
                                tsi_scaling,
                                n_lay_col);

                        //TOD fluxes to CPU
                        const int ncolgptsize = n_col*n_gpt*sizeof(Float);
                        cuda_safe_call(cudaMemcpy(sw_flux_dn_dir_inc_g,  sw_flux_dn_dir_inc.ptr(),  ncolgptsize, cudaMemcpyHostToDevice));
                        cuda_safe_call(cudaMemcpy(sw_flux_dn_dif_inc_g,  sw_flux_dn_dif_inc.ptr(),  ncolgptsize, cudaMemcpyHostToDevice));
                    }
                }

                if (is_day(this->mu0))
                {
                    exec_shortwave_rt(
                            thermo, timeloop, stats,
                            flux_up, flux_dn, flux_dn_dir, flux_net,
                            rt_flux_tod_dn, rt_flux_tod_up, rt_flux_sfc_dir, rt_flux_sfc_dif,
                            rt_flux_sfc_up, rt_flux_abs_dir, rt_flux_abs_dif,
                            t_lay_a, t_lev_a, h2o_a, clwp_a, ciwp_a,
                            compute_clouds);

                    calc_tendency_rt<<<gridGPU_3d, blockGPU_3d>>>(
                            fields.sd.at("thlt_rad")->fld_g,
                            rt_flux_abs_dir.ptr(), rt_flux_abs_dif.ptr(),
                            fields.rhoref_g, thermo.get_basestate_fld_g("exner"),
                            gd.dz_g,
                            gd.istart, gd.jstart, gd.kstart,
                            gd.iend, gd.jend, gd.kend,
                            gd.igc, gd.jgc, gd.kgc,
                            gd.icells, gd.ijcells,
                            gd.imax, gd.imax*gd.jmax);
                    cuda_check_error();

                    store_surface_fluxes_rt<<<gridGPU_2d, blockGPU_2d>>>(
                            sw_flux_up_sfc_g, sw_flux_dn_sfc_g,
                            rt_flux_sfc_dir.ptr(), rt_flux_sfc_dif.ptr(),
                            rt_flux_sfc_up.ptr(),
                            gd.istart, gd.iend,
                            gd.jstart, gd.jend,
                            gd.igc, gd.jgc,
                            gd.icells, gd.ijcells,
                            gd.imax);
                    cuda_check_error();
                }
                else
                {
                    // Set the surface fluxes to zero, for (e.g.) the land-surface model.
                    cudaMemset(sw_flux_dn_sfc_g, 0, gd.ijcells*sizeof(Float));
                    cudaMemset(sw_flux_up_sfc_g, 0, gd.ijcells*sizeof(Float));
                }

                if (do_radiation_stats)
                {
                    // Make sure that the top boundary is taken into account in case of fluxes.
                    auto do_gcs = [&](Field3d<Float>& out, const Array_gpu<Float,2>& in)
                    {
                        add_ghost_cells_g<<<gridGPU_3d, blockGPU_3d>>>(
                                out.fld_g, in.ptr(),
                                gd.istart, gd.jstart, gd.kstart,
                                gd.iend, gd.jend, gd.kend+1,
                                gd.icells, gd.ijcells,
                                gd.imax, gd.imax*gd.jmax);
                    };
                    auto do_gcs_rt = [&](Field3d<Float>& out, const Array_gpu<Float,3>& in)
                    {
                        add_ghost_cells_g<<<gridGPU_3d, blockGPU_3d>>>(
                                out.fld_g, in.ptr(),
                                gd.istart, gd.jstart, gd.kstart,
                                gd.iend, gd.jend, gd.kend,
                                gd.icells, gd.ijcells,
                                gd.imax, gd.imax*gd.jmax);
                    };

                    auto do_gcs_2d = [&](Float* out, const Array_gpu<Float,2>& in)
                    {
                        add_ghost_cells_2d_g<<<gridGPU_3d, blockGPU_3d>>>(
                                out, in.ptr(),
                                gd.istart, gd.jstart,
                                gd.iend, gd.jend,
                                gd.icells, gd.imax);
                    };

                    if (!is_day(this->mu0))
                    {
                        flux_up.fill(Float(0.));
                        flux_dn.fill(Float(0.));
                        flux_dn_dir.fill(Float(0.));
                    }

                    do_gcs(*fields.sd.at("sw_flux_up"), flux_up);
                    do_gcs(*fields.sd.at("sw_flux_dn"), flux_dn);
                    do_gcs(*fields.sd.at("sw_flux_dn_dir"), flux_dn_dir);

                    do_gcs_rt(*fields.sd.at("sw_heat_dir_rt"), rt_flux_abs_dir);
                    do_gcs_rt(*fields.sd.at("sw_heat_dif_rt"), rt_flux_abs_dif);

                    do_gcs_2d(sw_flux_sfc_dir_rt_g,rt_flux_sfc_dir);
                    do_gcs_2d(sw_flux_sfc_dif_rt_g,rt_flux_sfc_dif);
                    do_gcs_2d(sw_flux_sfc_up_rt_g,rt_flux_sfc_up);
                    do_gcs_2d(sw_flux_tod_dn_rt_g,rt_flux_tod_dn);
                    do_gcs_2d(sw_flux_tod_up_rt_g,rt_flux_tod_up);

                    // clear sky
                    if (sw_clear_sky_stats)
                    {
                        if (is_day(this->mu0))
                        {
                            exec_shortwave(
                                    thermo, timeloop, stats,
                                    flux_up, flux_dn, flux_dn_dir, flux_net,
                                    t_lay_a, t_lev_a, h2o_a, clwp_a, ciwp_a,
                                    !compute_clouds);
                        }
                        do_gcs(*fields.sd.at("sw_flux_up_clear"), flux_up);
                        do_gcs(*fields.sd.at("sw_flux_dn_clear"), flux_dn);
                        do_gcs(*fields.sd.at("sw_flux_dn_dir_clear"), flux_dn_dir);
                    }
                 }
             }
         } // End try block.
         catch (std::exception& e)
         {
             #ifdef USEMPI
            std::cout << "SINGLE PROCESS EXCEPTION: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            #else
            throw;
            #endif
        }

        fields.release_tmp_g(t_lay);
        fields.release_tmp_g(t_lev);
        fields.release_tmp_g(h2o);
        fields.release_tmp_g(clwp);
        fields.release_tmp_g(ciwp);
    }

    // Always add the tendency.
    add_tendency<<<gridGPU_3d, blockGPU_3d>>>(
            fields.st.at("thl")->fld_g,
            fields.sd.at("thlt_rad")->fld_g,
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            gd.icells, gd.ijcells);
    cuda_check_error();

    cudaDeviceSynchronize();
    stats.calc_tend(*fields.st.at("thl"), tend_name);
}


template <typename TF>
std::vector<TF>& Radiation_rrtmgp_rt<TF>::get_surface_radiation(const std::string& name)
{
    throw std::runtime_error("Radiation_rrtmgp_rt is not implemented yet on the GPU");
}


template <typename TF>
TF* Radiation_rrtmgp_rt<TF>::get_surface_radiation_g(const std::string& name)
{
    if (name == "sw_down")
        return sw_flux_dn_sfc_g;
    else if (name == "sw_up")
        return sw_flux_up_sfc_g;
    else if (name == "lw_down")
        return lw_flux_dn_sfc_g;
    else if (name == "lw_up")
        return lw_flux_up_sfc_g;
    else
    {
        std::string error = "Variable \"" + name + "\" is not a valid surface radiation field";
        throw std::runtime_error(error);
    }
}


template <typename TF>
void Radiation_rrtmgp_rt<TF>::clear_device()
{
    cuda_safe_call(cudaFree(lw_flux_dn_sfc_g));
    cuda_safe_call(cudaFree(lw_flux_up_sfc_g));
    cuda_safe_call(cudaFree(sw_flux_dn_sfc_g));
    cuda_safe_call(cudaFree(sw_flux_up_sfc_g));

    cuda_safe_call(cudaFree(sw_flux_sfc_dir_rt_g));
    cuda_safe_call(cudaFree(sw_flux_sfc_dif_rt_g));
    cuda_safe_call(cudaFree(sw_flux_sfc_up_rt_g));
    cuda_safe_call(cudaFree(sw_flux_tod_dn_rt_g));
    cuda_safe_call(cudaFree(sw_flux_tod_up_rt_g));

    cuda_safe_call(cudaFree(lw_flux_dn_inc_g));
    cuda_safe_call(cudaFree(sw_flux_dn_dir_inc_g));
    cuda_safe_call(cudaFree(sw_flux_dn_dif_inc_g));
}


template<typename TF>
void Radiation_rrtmgp_rt<TF>::exec_all_stats(
        Stats<TF>& stats, Cross<TF>& cross,
        Dump<TF>& dump, Column<TF>& column,
        Thermo<TF>& thermo, Timeloop<TF>& timeloop,
        const unsigned long itime, const int iotime)
{
    const bool do_stats  = stats.do_statistics(itime);
    const bool do_cross  = cross.do_cross(itime) && crosslist.size() > 0;
    const bool do_column = column.do_column(itime);

    // Return in case of no stats or cross section.
    if ( !(do_stats || do_cross || do_column) )
        return;

    const Float no_offset = 0.;
    const Float no_threshold = 0.;

    // CvH: lots of code repetition with exec()
    auto& gd = grid.get_grid_data();

    const bool compute_clouds = true;

    // Use a lambda function to avoid code repetition.
    auto save_stats_and_cross = [&](
            Field3d<TF>& array, const std::string& name, const std::array<int,3>& loc)
    {
        if (do_stats)
            stats.calc_stats(name, array, no_offset, no_threshold);

        if (do_cross)
        {
            if (std::find(crosslist.begin(), crosslist.end(), name) != crosslist.end())
                cross.cross_simple(array.fld.data(), name, iotime, loc);
        }

        if (do_column)
            column.calc_column(name, array.fld.data(), no_offset);
    };

    if (sw_longwave)
    {
        save_stats_and_cross(*fields.sd.at("lw_flux_up"), "lw_flux_up", gd.wloc);
        save_stats_and_cross(*fields.sd.at("lw_flux_dn"), "lw_flux_dn", gd.wloc);

        if (sw_clear_sky_stats)
        {
            save_stats_and_cross(*fields.sd.at("lw_flux_up_clear"), "lw_flux_up_clear", gd.wloc);
            save_stats_and_cross(*fields.sd.at("lw_flux_dn_clear"), "lw_flux_dn_clear", gd.wloc);
        }
    }

    if (sw_shortwave)
    {
        save_stats_and_cross(*fields.sd.at("sw_flux_up"), "sw_flux_up", gd.wloc);
        save_stats_and_cross(*fields.sd.at("sw_flux_dn"), "sw_flux_dn", gd.wloc);
        save_stats_and_cross(*fields.sd.at("sw_flux_dn_dir"), "sw_flux_dn_dir", gd.wloc);

        save_stats_and_cross(*fields.sd.at("sw_heat_dir_rt"), "sw_heat_dir_rt", gd.sloc);
        save_stats_and_cross(*fields.sd.at("sw_heat_dif_rt"), "sw_heat_dif_rt", gd.sloc);

        if (sw_clear_sky_stats)
        {
            save_stats_and_cross(*fields.sd.at("sw_flux_up_clear"), "sw_flux_up_clear", gd.wloc);
            save_stats_and_cross(*fields.sd.at("sw_flux_dn_clear"), "sw_flux_dn_clear", gd.wloc);
            save_stats_and_cross(*fields.sd.at("sw_flux_dn_dir_clear"), "sw_flux_dn_dir_clear", gd.wloc);
        }

        const int nsfcsize = gd.ijcells*sizeof(Float);
        cuda_safe_call(cudaMemcpy(sw_flux_sfc_dir_rt.data(), sw_flux_sfc_dir_rt_g, nsfcsize, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(sw_flux_sfc_dif_rt.data(), sw_flux_sfc_dif_rt_g, nsfcsize, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(sw_flux_sfc_up_rt.data(), sw_flux_sfc_up_rt_g, nsfcsize, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(sw_flux_tod_dn_rt.data(), sw_flux_tod_dn_rt_g, nsfcsize, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(sw_flux_tod_up_rt.data(), sw_flux_tod_up_rt_g, nsfcsize, cudaMemcpyDeviceToHost));

        if (do_stats)
        {
            stats.calc_stats_2d("sw_flux_sfc_dir_rt", sw_flux_sfc_dir_rt, no_offset);
            stats.calc_stats_2d("sw_flux_sfc_dif_rt", sw_flux_sfc_dif_rt, no_offset);
            stats.calc_stats_2d("sw_flux_sfc_up_rt", sw_flux_sfc_up_rt, no_offset);
            stats.calc_stats_2d("sw_flux_tod_dn_rt", sw_flux_tod_dn_rt, no_offset);
            stats.calc_stats_2d("sw_flux_tod_up_rt", sw_flux_tod_up_rt, no_offset);
        }

        if (do_cross)
        {
            if (std::find(crosslist.begin(), crosslist.end(), "sw_flux_sfc_dir_rt") != crosslist.end())
                cross.cross_plane(sw_flux_sfc_dir_rt.data(), "sw_flux_sfc_dir_rt", iotime);
            if (std::find(crosslist.begin(), crosslist.end(), "sw_flux_sfc_dif_rt") != crosslist.end())
                cross.cross_plane(sw_flux_sfc_dif_rt.data(), "sw_flux_sfc_dif_rt", iotime);
            if (std::find(crosslist.begin(), crosslist.end(), "sw_flux_sfc_up_rt") != crosslist.end())
                cross.cross_plane(sw_flux_sfc_up_rt.data(), "sw_flux_sfc_up_rt", iotime);
            if (std::find(crosslist.begin(), crosslist.end(), "sw_flux_tod_dn_rt") != crosslist.end())
                cross.cross_plane(sw_flux_tod_dn_rt.data(), "sw_flux_tod_dn_rt", iotime);
            if (std::find(crosslist.begin(), crosslist.end(), "sw_flux_tod_up_rt") != crosslist.end())
                cross.cross_plane(sw_flux_tod_up_rt.data(), "sw_flux_tod_up_rt", iotime);
        }
    }

    stats.set_time_series("sza", std::acos(mu0));
    stats.set_time_series("saa", azimuth);
    stats.set_time_series("tsi_scaling", this->tsi_scaling);
    stats.set_time_series("sw_flux_dn_toa", sw_flux_dn_col({1,n_lev_col}));
}
#endif

#ifdef FLOAT_SINGLE
template class Radiation_rrtmgp_rt<float>;
#else
template class Radiation_rrtmgp_rt<double>;
#endif