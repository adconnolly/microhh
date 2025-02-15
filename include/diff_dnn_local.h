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

#ifndef DIFF_DNN_LOCAL_H
#define DIFF_DNN_LOCAL_H

#include "diff.h"
#include "boundary_cyclic.h"
#include "field3d_operators.h"
#include <torch/torch.h>
#include <torch/script.h>

template<typename> class Stats;


template<typename TF>
class Diff_dnn_local : public Diff<TF>
{
    public:
        Diff_dnn_local(Master&, Grid<TF>&, Fields<TF>&, Boundary<TF>&, Input&);
        ~Diff_dnn_local();

        Diffusion_type get_switch() const;
        unsigned long get_time_limit(unsigned long, double);
        double get_dn(double);

        void create(Stats<TF>&);
        void init();
        void exec(Stats<TF>&);
        void exec_viscosity(Thermo<TF>&);
        void diff_flux(Field3d<TF>&, const Field3d<TF>&);
        void exec_stats(Stats<TF>&);

        #ifdef USECUDA
        void prepare_device(Boundary<TF>&);
        void clear_device();
        #endif

    private:
        using Diff<TF>::master;
        using Diff<TF>::grid;
        using Diff<TF>::fields;
        using Diff<TF>::boundary;
        Boundary_cyclic<TF> boundary_cyclic;
        Field3d_operators<TF> field3d_operators;

        using Diff<TF>::tPr;

        const Diffusion_type swdiff = Diffusion_type::Diff_dnn_local;

        void create_stats(Stats<TF>&);

        TF* mlen_g;

        double dnmax;
        double dnmul;

        double cs;        
        
        double ce; // for the TKE based estimate of eddy viscosity, not using currently
        TF lgrid;
       
        //torch::Tensor Tau;
        at::Tensor Tau;
        bool swdeviatoric;
        std::string dnnpath;
        torch::jit::script::Module dnn;
       
        const std::string tend_name = "diff";
        const std::string tend_longname = "Diffusion";
};
#endif
