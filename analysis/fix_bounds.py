import numpy as np

def fix_bounds(u_string,coarse,itime,noslip=False,fix_bad_pad=False,nzbuffer=2):
    
    # leave out last point b/c it was bad padding from coarse graining
    # and ignored during training, but 
    # include sfc BC for interp, so shape stays the same    
    z_coarse = coarse.variables["z"]
    z_interp=np.zeros(z_coarse.shape)
    z_interp[1:]=z_coarse[:-1]
    
    u_coarse=np.array(np.mean(coarse.variables[u_string][:,:,:,itime],axis=(1,2)))
    u=np.zeros(u_coarse.shape[0])
    u[1:]=u_coarse[:-1]
    if not noslip:
        u[0]=3*u[1]-2*u[2] # extrap to surface, e.g. for b
    
    if fix_bad_pad:
        du=u[-nzbuffer-1]-u[-nzbuffer-2]
        for iz in range(-nzbuffer,0,1):
            u[iz]=u[-nzbuffer-1]+(nzbuffer+iz+1)*du

    return u,z_interp