# NCAR Cheyenne, GNU compilers for LibTorch
#
# module load gnu/10.1.0 
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
set(BOOST_INCLUDE_DIR  "/glade/u/apps/ch/opt/boost/1_67_0/include")
set(FFTW_INCLUDE_DIR   "/glade/u/apps/ch/opt/fftw/3.3.8/gnu/10.1.0/include")
set(FFTW_LIB           "/glade/u/apps/ch/opt/fftw/3.3.8/gnu/10.1.0/lib/libfftw3.a")
set(NETCDF_INCLUDE_DIR "/glade/u/apps/ch/opt/netcdf/4.8.1/gnu/10.1.0/include")
set(NETCDF_LIB_C       "/glade/u/apps/ch/opt/netcdf/4.8.1/gnu/10.1.0/lib/libnetcdf.a")
set(NETCDF_LIB_CPP     "/glade/u/apps/ch/opt/netcdf/4.8.1/gnu/10.1.0/lib/libnetcdf_c++4.a")
set(HDF5_LIB_1         "/glade/u/apps/ch/opt/netcdf/4.8.1/gnu/10.1.0/lib/libhdf5.a")
set(HDF5_LIB_2         "/glade/u/apps/ch/opt/netcdf/4.8.1/gnu/10.1.0/lib/libhdf5_hl.a")
set(SZIP_LIB           "/glade/u/apps/ch/opt/netcdf/4.8.1/gnu/10.1.0/lib/libsz.a")
set(LIBS ${FFTW_LIB} ${NETCDF_LIB_CPP} ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${SZIP_LIB} m z curl)
set(INCLUDE_DIRS ${BOOST_INCLUDE_DIR} ${FFTW_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR})

add_definitions(-DRESTRICTKEYWORD=__restrict__)
