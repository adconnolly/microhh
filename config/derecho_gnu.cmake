# NCAR Derecho, GNU compilers for LibTorch
#
# module load gcc/13.2.0 
#       
if(USEMPI) 
  set(ENV{CC}  mpicc)  # C compiler for parallel build
  set(ENV{CXX} mpicxx) # C++ compiler for parallel build
else()
  set(ENV{CC}  gcc ) # C compiler for serial build 
  set(ENV{CXX} g++) # C++ compiler for serial build
endif()

set(USER_CXX_FLAGS "-std=c++17 -march=native") # -gxx-name=/glade/u/apps/ch/opt/gnu/10.1.0/bin/gcc")
set(USER_CXX_FLAGS_RELEASE "-O3")
set(USER_CXX_FLAGS_DEBUG "-g -O0")
set(BOOST_INCLUDE_DIR  "/glade/u/apps/derecho/23.09/spack/opt/spack/boost/1.83.0/gcc/12.2.0/zdhe/include/")
set(FFTW_INCLUDE_DIR   "/glade/u/apps/derecho/23.09/spack/opt/spack/fftw/3.3.10/gcc/13.2.0/7sd3/include/")
set(FFTW_LIB           "/glade/u/apps/derecho/23.09/spack/opt/spack/fftw/3.3.10/gcc/13.2.0/7sd3/lib/libfftw3.a")
set(NETCDF_INCLUDE_DIR "/glade/u/apps/derecho/23.09/spack/opt/spack/netcdf/4.9.2/gcc/13.2.0/zywl/include/")
set(NETCDF_LIB_C       "/glade/u/apps/derecho/23.09/spack/opt/spack/netcdf/4.9.2/gcc/13.2.0/zywl/lib/libnetcdf.a")
set(NETCDF_LIB_CPP     "/glade/u/apps/derecho/23.09/spack/opt/spack/netcdf/4.9.2/gcc/13.2.0/zywl/lib/libnetcdf_c++4.a")
set(HDF5_LIB_1         "/glade/u/apps/derecho/23.09/spack/opt/spack/hdf5/1.14.3/gcc/13.2.0/ten6/lib/libhdf5.a")
set(HDF5_LIB_2         "/glade/u/apps/derecho/23.09/spack/opt/spack/hdf5/1.14.3/gcc/13.2.0/ten6/lib/libhdf5_hl.a")
set(SZIP_LIB           "/glade/u/apps/derecho/23.09/spack/opt/spack/libszip/2.1.1/gcc/7.5.0/cg5o/lib/libsz.a")
set(XML_LIB           "/glade/u/apps/derecho/23.09/spack/opt/spack/libxml2/2.10.3/gcc/7.5.0/p2qi/lib/libxml2.so")
set(LIBS ${FFTW_LIB} ${NETCDF_LIB_CPP} ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${SZIP_LIB} ${XML_LIB} m z curl)
set(INCLUDE_DIRS ${BOOST_INCLUDE_DIR} ${FFTW_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR})

add_definitions(-DRESTRICTKEYWORD=__restrict__)
