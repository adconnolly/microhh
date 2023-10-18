#Re1800:
import numpy as np
#from pylab import *
from scipy import special
import netCDF4 as nc


# set the height
#kmax  = 3328
kmax = 6
zsize = 18.074844397670482
dz = zsize / kmax

a_star = 5/8*zsize
theta_ref = 300
g         = 9.8
dtheta_dz_z0 =0.028941114424258
theta_z = np.zeros(kmax)
theta_0 = 299.8649820802043
b = np.zeros(kmax)

z = np.zeros(kmax)
u = np.zeros(kmax)
v = np.zeros(kmax)
s = np.zeros(kmax)
ug = np.zeros(kmax)
vg = np.zeros(kmax)


ug[:] = 0.049295030175465
vg[:] = 0.

z = np.linspace(0.5*dz, zsize-0.5*dz, kmax)

visc = 0.000015
fc   = 0.0001
gamma = (fc / (2.*visc))**.5

u[:] = ug[:]
v[:] = vg[:]

# obtain theta_z and b for microhh
for k in range(kmax):
   theta_z[k]=a_star/2*(np.power(-np.pi/np.log(0.01),0.5))*dtheta_dz_z0*special.erf(z[k]/a_star/np.power(-np.log(0.01),-0.5))+theta_0
   s[k]=g*(theta_z[k]-theta_ref)/theta_ref
   b[k]=s[k]
# analytical solution as the starting profile to reduce run time
#for k in range(kmax):
#  u[k] = ug[k]*(1. - exp(-gamma*z[k]) * cos(gamma*z[k]))
#  v[k] = ug[k]*(     exp(-gamma*z[k]) * sin(gamma*z[k]))

# write the data to a file
#proffile = open('ekman.prof','w')
#proffile.write('{0:^20s} {1:^20s} {2:^20s} {3:^20s} {4:^20s} {5:^20s} {6:^20s}\n'.format('z','u','v','ug','vg','s','b'))
#for k in range(kmax):
#  proffile.write('{0:1.14E} {1:1.14E} {2:1.14E} {3:1.14E} {4:1.14E} {5:1.14E} {6:1.14E}\n'.format(z[k], u[k], v[k], ug[k], vg[k], s[k],b[k]))
#proffile.close()

float_type = 'f8'

nc_file = nc.Dataset("SBL1800_input.nc", mode="w", datamodel="NETCDF4", clobber=False)

nc_file.createDimension("z", kmax)
nc_z  = nc_file.createVariable("z" , float_type, ("z"))

nc_group_init = nc_file.createGroup("init");
nc_u = nc_group_init.createVariable("u", float_type, ("z"))
nc_v = nc_group_init.createVariable("v", float_type, ("z"))
nc_u_geo = nc_group_init.createVariable("u_geo", float_type, ("z"))
nc_v_geo = nc_group_init.createVariable("v_geo", float_type, ("z"))
nc_s = nc_group_init.createVariable("s", float_type, ("z"))
nc_b = nc_group_init.createVariable("b", float_type, ("z"))

nc_z[:] = z[:]
nc_u[:] = u[:]
nc_v[:] = v[:]
nc_u_geo[:] = ug[:]
nc_v_geo[:] = vg[:]
nc_s[:] = s[:]
nc_b[:] = b[:]

nc_file.close()
