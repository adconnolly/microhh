import numpy as np
import xarray as xr
import torch
from e2cnn import nn
from e2cnn import gspaces
import gc

def preprocess(files,filemaskpercents,scalingStr,fileUgs=None,fileRes=None,fileB0s=None,size=3,irun='', reshape=True):
    # Any of the below could be changed to inputs of the function
    path="/glade/u/home/adac/work/DNStoLES/coarseData/"
    #size = 1 # number of neighboring points to include in each input sample
    size=int((size-1)/2)
    vsize=1

    yList=list()
    xList=list()
    tauScaleList=list()
    maskDict = {}
    for ifile in range(len(files)): #tdqm notebook just draws progress bar
        file=files[ifile]
        ds=xr.open_dataset(path+file,decode_times=0)
        print(ds)

        # Getting input variables, removing laminar layers, wrapping periodic varaibles as reordering to x,y,z,t 
        b=np.transpose(np.pad(cut_laminar(ds['b'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        u=np.transpose(np.pad(cut_laminar(ds['u'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap') , [2, 1, 0,3])
        v=np.transpose(np.pad(cut_laminar(ds['v'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        w=np.transpose(np.pad(cut_laminar(ds['w'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        inputFields = np.array([u, v, w, b])
        
        # Getting output fields
        tau_12 = np.transpose(cut_laminar(ds['tau12'].values), [2, 1, 0,3])
        tau_13 = np.transpose(cut_laminar(ds['tau13'].values), [2, 1, 0,3])
        tau_23 = np.transpose(cut_laminar(ds['tau23'].values), [2, 1, 0,3])
        tau_11 = np.transpose(cut_laminar(ds['tau11'].values), [2, 1, 0,3])
        tau_22 = np.transpose(cut_laminar(ds['tau22'].values), [2, 1, 0,3])
        tau_33 = np.transpose(cut_laminar(ds['tau33'].values), [2, 1, 0,3])
        outputFields = np.array([tau_11,tau_12, tau_13, tau_22, tau_23, tau_33])

        if scalingStr=='local':
            # Getting grid vairables to comput grid size
            x=ds['x'].values 
            yy=ds['y'].values
            z=cut_laminar(ds['z'].values)
            dx=np.mean(np.diff(x))
            dy=np.mean(np.diff(yy))
            dz=np.mean(np.diff(z))
            #delta=(dx*dy*dz)**(1.0/3.0)

            # Computing gradients for scaling saling variables: TKE, TPE, and N2
            d11=(u[2:,1:-1,:,:]-u[:-2,1:-1,:,:])/(2*dx)
            d12=(u[1:-1,2:,:,:]-u[1:-1,:-2,:,:])/(2*dy)
            d13=np.gradient(u[1:-1,1:-1,:,:],dz,axis=2)

            d21=(v[2:,1:-1,:,:]-v[:-2,1:-1,:,:])/(2*dx)
            d22=(v[1:-1,2:,:,:]-v[1:-1,:-2,:,:])/(2*dy)
            d23=np.gradient(v[1:-1,1:-1,:,:],dz,axis=2)

            d31=(w[2:,1:-1,:,:]-w[:-2,1:-1,:,:])/(2*dx)
            d32=(w[1:-1,2:,:,:]-w[1:-1,:-2,:,:])/(2*dy)
            d33=np.gradient(w[1:-1,1:-1,:,:],dz,axis=2)

            #dijdij = (d11*d11 + d12*d12 + d13*d13 + d21*d21 + d22*d22 + d23*d23 + d31*d31 + d32*d32 + d33*d33)
            
            TKE_h = dx**2*(d11**2 + d21**2) + dy**2*(d12**2 + d22**2) + dz**2*(d13**2 + d23**2) 
            TKE_v = dx**2*d31**2 + dy**2*d32**2 + dz**2*d33**2
            TKE = TKE_h + TKE_v
            #TKE = (delta**2/1.0)*dijdij
            del d11,d12,d13,d21,d22,d23,d31,d32,d33,TKE_h

            db1=(b[2:,1:-1,:,:]-b[:-2,1:-1,:,:])/(2*dx)
            db2=(b[1:-1,2:,:,:]-b[1:-1,:-2,:,:])/(2*dy)
            db3=np.gradient(b[1:-1,1:-1,:,:],dz,axis=2)
            N2=db3
            #dbkdbk=db1*db1+db2*db2+db3*db3
            TPE=(dx**2*db1**2+dy**2*db2**2+dz**2*db3**2)/N2
            #TPE = (delta**2/1.0)*dbkdbk/N2
            del db1,db2,db3 
            gc.collect()

            hvelScale = np.pad(np.sqrt(TKE),((size,size),(size,size),(0,0),(0,0)),mode='wrap')
            vvelScale = np.pad(np.sqrt(TKE_v),((size,size),(size,size),(0,0),(0,0)),mode='wrap')
            bScale = np.pad(TPE/dz,((size,size),(size,size),(0,0),(0,0)),mode='wrap')
            ti3Scale=np.sqrt(TKE*TKE_v)
            tijScale=np.array([TKE,TKE,ti3Scale,TKE,ti3Scale,TKE_v])
        
        elif scalingStr=='global':
            Ug=fileUgs[ifile]
            Re=fileRes[ifile]
            b0=fileB0s[ifile]
            #g=9.8
            hvelScale = Ug/np.sqrt(Re)*np.ones(u.shape)
            vvelScale = Ug/np.sqrt(Re)*np.ones(u.shape)
            #bScale = g*np.ones(u.shape)
            bScale = -b0*np.ones(u.shape)
            tijScale=Ug**2/Re*np.ones(outputFields.shape)

        elif scalingStr=='unscaled':
            reshape=False
            hvelScale = np.ones(u.shape)
            vvelScale = np.ones(u.shape)
            bScale = np.ones(u.shape)
            tijScale=np.ones(outputFields.shape)

        nx=outputFields.shape[1]
        ny=outputFields.shape[2]
        nz=outputFields.shape[3]
        nt=outputFields.shape[4]
        mask = np.random.rand(nx,ny,nz,nt) < filemaskpercents[ifile]
        maskDict["mask_"+file+'_'+str(irun)]=mask
        
        for i in range(size, inputFields.shape[1] - size):
            for j in range(size, inputFields.shape[2] - size):
                for k in range(vsize, inputFields.shape[3] - vsize):
                    for it in range(inputFields.shape[4]):
                        if mask[i-size,j-size,k,it]:
                            scaledInput=[scale(inputFields[0,i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it],sd=hvelScale[i, j, k, it]),
                                            scale(inputFields[1,i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it],sd=hvelScale[i, j, k, it]),
                                            scale(inputFields[2,i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it],sd=vvelScale[i, j, k, it]),
                                            scale(inputFields[3,i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it],sd=bScale[i, j, k, it])]
                            xList.append(scaledInput) 
                            yList.append(outputFields[:,i-size,j-size,k,it])
                            tauScaleList.append(tijScale[:,i-size,j-size,k,it])
                                 
    y=np.array(yList)
    del yList
    x3d=np.array(xList)
    del xList
    tauScale=np.array(tauScaleList)
    del tauScaleList
    gc.collect()

    print("output shape is "+str(y.shape))
    print("input shape should be "+str(x3d.shape))
    if reshape:
        size=int(2*size+1)
        x=myreshape(x3d,size=size)                    
        print("input shape to do 3rd dimension as channel in R2Conv is "+str(x.shape))
    else:
        x=x3d
        
    return x, y, tauScale, maskDict

class CNDNN(torch.nn.Module):
    def __init__(self,Nhid,N,size,device):
        super(CNDNN, self).__init__()
        
        r2_act = gspaces.Rot2dOnR2(N = N)

        self.feat_type_in = nn.FieldType(r2_act, 3*[r2_act.irrep(1)]+2*3*[r2_act.trivial_repr])
        self.feat_type_hid1 = nn.FieldType(r2_act, Nhid[0]*[r2_act.regular_repr])
        self.feat_type_hid2 = nn.FieldType(r2_act, Nhid[1]*[r2_act.regular_repr])
        self.feat_type_hid3 = nn.FieldType(r2_act, Nhid[2]*[r2_act.regular_repr])
        self.feat_type_hid4 = nn.FieldType(r2_act, Nhid[3]*[r2_act.regular_repr])
        if N==4:
            self.feat_type_out = nn.FieldType(r2_act, 2*[r2_act.irrep(0)]+[r2_act.irrep(1)]+2*[r2_act.irrep(2)])
        else:
            self.feat_type_out = nn.FieldType(r2_act, 2*[r2_act.irrep(0)]+[r2_act.irrep(1)]+[r2_act.irrep(2)])

        
        self.input_layer = nn.R2Conv(self.feat_type_in, self.feat_type_hid1, kernel_size=size,bias=False)
        self.relu1 = nn.ReLU(self.feat_type_hid1)
        self.hid1 = nn.R2Conv(self.feat_type_hid1, self.feat_type_hid2, kernel_size=1,bias=False)
        self.relu2 = nn.ReLU(self.feat_type_hid2)
        self.hid2 = nn.R2Conv(self.feat_type_hid2, self.feat_type_hid3, kernel_size=1,bias=False)
        self.relu3 = nn.ReLU(self.feat_type_hid3)
        self.hid3 = nn.R2Conv(self.feat_type_hid3, self.feat_type_hid4, kernel_size=1,bias=False)
        self.relu4 = nn.ReLU(self.feat_type_hid4)
        self.hid4 = nn.R2Conv(self.feat_type_hid4, self.feat_type_out, kernel_size=1,bias=False)
   
        self.Pinv=torch.tensor([[1/2, 0, 0, 0, -1/4, 1/4],
                           [0, 0, 0, 0, 1/4, 1/4],
                           [0, 0, 1, 0, 0, 0],
                           [1/2, 0, 0, 0, 1/4, -1/4],
                           [0, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0, 0]]).float().to(device)
        self.change_basis=torch.nn.Linear(6,6, bias=False)
        self.change_basis.weight=torch.nn.Parameter(self.Pinv, requires_grad=False)

    def forward(self,x):
        

        x = nn.GeometricTensor(x, self.feat_type_in)

        x= self.input_layer(x)
        x = self.relu1(x)
        x = self.hid1(x)
        x = self.relu2(x)
        x = self.hid2(x)
        x = self.relu3(x)
        x = self.hid3(x)
        x = self.relu4(x)
        x = self.hid4(x)
        x = x.tensor.squeeze(-1).squeeze(-1)
        x = self.change_basis(x)
        
        return x


def train_model(model, criterion, loader, optimizer, scheduler, weights, device):
    model.train()
    for step, (x_batch, y_batch, tauScale_batch) in enumerate(loader):  # for each training step
        x_batch=x_batch.to(device)
        y_batch=y_batch.to(device)
        tauScale_batch=tauScale_batch.to(device)
        
        prediction = model(x_batch)

        loss = criterion(prediction*tauScale_batch*weights, y_batch*weights).cpu()
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients to update weights
    scheduler.step()

def test_model(model, criterion, trainloader, weights, device, text = 'validation'):
    model.eval() # Evaluation mode (important when having dropout layers)
    test_loss = 0
    with torch.no_grad():
        for step, (x_batch, y_batch, tauScale_batch) in enumerate(trainloader):  # for each training step
            x_batch=x_batch.to(device)
            y_batch=y_batch.to(device)
            tauScale_batch=tauScale_batch.to(device)
        
            prediction = model(x_batch)

            loss = criterion(prediction*tauScale_batch*weights, y_batch*weights).cpu()
            test_loss = test_loss + loss.data.numpy() # Keep track of the loss 
        test_loss /= len(trainloader) # dividing by the number of batches
        print(text + ' loss:',test_loss)
    return test_loss

def cut_laminar(x):
    # Cuts the input 75% along the z-axis to remove laminar layers
    return x[:int(0.75*x.shape[0])]

def scale(x,mean=None,sd=None):

    if mean==None:
        mean=np.mean(x)
        
    if sd==None:
        sd=np.std(x)
    
    return (x - mean) / sd

def myreshape(xtest,nvars=4,size=3):
    #Resulting order is, with each entry being a 3*3 horiz. plane:
    #u(k=0),v(k=0),u(k=1),v(k=1),u(k=2),v(k=2),w(k=0),w(k=1),w(k=2),b(k=0),b(k=1),b(k=2)
    xnew=np.reshape(xtest.copy(),(xtest.shape[0],nvars*3,size,size))
    # Have to make this loop explicit because grouping u,v as 
    #2d vector is inherent to using irrep(1) representation type
    for v in range(2):
        for k in range(xtest.shape[-1]):
            xnew[:,2*k+v,:,:]=xtest[:,v,:,:,k]
    if nvars>2:   
        for v in range(2,nvars):
            for k in range(xtest.shape[-1]):
                xnew[:,3*v+k,:,:]=xtest[:,v,:,:,k]
        
    return xnew

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, val_loss):
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class baselineDNN(torch.nn.Module):
    def __init__(self, input_shape,Nl):
        super(baselineDNN, self).__init__()
                            
        self.conv = torch.nn.Conv2d(input_shape[0], Nl[0], (input_shape[1],input_shape[2]),padding='valid',bias=False)
        self.linear1 = torch.nn.Linear(Nl[0],Nl[1],bias=False)
        self.linear2 = torch.nn.Linear(Nl[1],Nl[2],bias=False)
        self.linear3 = torch.nn.Linear(Nl[2],Nl[3],bias=False)
        self.linear4 = torch.nn.Linear(Nl[3],6,bias=False)
                                                                    
    def forward(self,x):
        x=torch.relu(self.conv(x))
        x=x.view(x.size(0), -1)
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=torch.relu(self.linear3(x))
        x=self.linear4(x)

        return x.squeeze()

def preprocess_dataAug(files, filemaskpercents, scalingStr, fileUgs=None, fileRes=None, fileB0s=None, size=3, irun='', krotAll = [0,1,2,3], maskdict=None, reshape=True):
    # Any of the below could be changed to inputs of the function
    path="/burg/glab/users/ac5006/DNStoLES/coarseData/fullTau/"
    #size = 1 # number of neighboring points to include in each input sample
    size=int((size-1)/2)
    vsize=1

    yList=list()
    xList=list()
    tauScaleList=list()
    maskDict = {}
    for ifile in range(len(files)): #tdqm notebook just draws progress bar
        file=files[ifile]
        ds=xr.open_dataset(path+file,decode_times=0)
        print(ds)

        # Getting input variables, removing laminar layers, wrapping periodic varaibles as reordering to x,y,z,t 
        b=np.transpose(np.pad(cut_laminar(ds['b'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        u=np.transpose(np.pad(cut_laminar(ds['u'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap') , [2, 1, 0,3])
        v=np.transpose(np.pad(cut_laminar(ds['v'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        w=np.transpose(np.pad(cut_laminar(ds['w'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        inputFields = np.array([u, v, w, b])
        
        # Getting output fields
        tau_12 = np.transpose(cut_laminar(ds['tau12'].values), [2, 1, 0,3])
        tau_13 = np.transpose(cut_laminar(ds['tau13'].values), [2, 1, 0,3])
        tau_23 = np.transpose(cut_laminar(ds['tau23'].values), [2, 1, 0,3])
        tau_11 = np.transpose(cut_laminar(ds['tau11'].values), [2, 1, 0,3])
        tau_22 = np.transpose(cut_laminar(ds['tau22'].values), [2, 1, 0,3])
        tau_33 = np.transpose(cut_laminar(ds['tau33'].values), [2, 1, 0,3])
        outputFields = np.array([tau_11,tau_12, tau_13, tau_22, tau_23, tau_33])

        if scalingStr=='local':
            # Getting grid vairables to comput grid size
            x=ds['x'].values 
            yy=ds['y'].values
            z=cut_laminar(ds['z'].values)
            dx=np.mean(np.diff(x))
            dy=np.mean(np.diff(yy))
            dz=np.mean(np.diff(z))
            #delta=(dx*dy*dz)**(1.0/3.0)

            # Computing gradients for scaling saling variables: TKE, TPE, and N2
            d11=(u[2:,1:-1,:,:]-u[:-2,1:-1,:,:])/(2*dx)
            d12=(u[1:-1,2:,:,:]-u[1:-1,:-2,:,:])/(2*dy)
            d13=np.gradient(u[1:-1,1:-1,:,:],dz,axis=2)

            d21=(v[2:,1:-1,:,:]-v[:-2,1:-1,:,:])/(2*dx)
            d22=(v[1:-1,2:,:,:]-v[1:-1,:-2,:,:])/(2*dy)
            d23=np.gradient(v[1:-1,1:-1,:,:],dz,axis=2)

            d31=(w[2:,1:-1,:,:]-w[:-2,1:-1,:,:])/(2*dx)
            d32=(w[1:-1,2:,:,:]-w[1:-1,:-2,:,:])/(2*dy)
            d33=np.gradient(w[1:-1,1:-1,:,:],dz,axis=2)

            #dijdij = (d11*d11 + d12*d12 + d13*d13 + d21*d21 + d22*d22 + d23*d23 + d31*d31 + d32*d32 + d33*d33)
            
            TKE_h = dx**2*(d11**2 + d21**2) + dy**2*(d12**2 + d22**2) + dz**2*(d13**2 + d23**2) 
            TKE_v = dx**2*d31**2 + dy**2*d32**2 + dz**2*d33**2
            TKE = TKE_h + TKE_v
            #TKE = (delta**2/1.0)*dijdij
            del d11,d12,d13,d21,d22,d23,d31,d32,d33,TKE_h

            db1=(b[2:,1:-1,:,:]-b[:-2,1:-1,:,:])/(2*dx)
            db2=(b[1:-1,2:,:,:]-b[1:-1,:-2,:,:])/(2*dy)
            db3=np.gradient(b[1:-1,1:-1,:,:],dz,axis=2)
            N2=db3
            #dbkdbk=db1*db1+db2*db2+db3*db3
            TPE=(dx**2*db1**2+dy**2*db2**2+dz**2*db3**2)/N2
            #TPE = (delta**2/1.0)*dbkdbk/N2
            del db1,db2,db3 
            gc.collect()

            hvelScale = np.pad(np.sqrt(TKE),((size,size),(size,size),(0,0),(0,0)),mode='wrap')
            vvelScale = np.pad(np.sqrt(TKE_v),((size,size),(size,size),(0,0),(0,0)),mode='wrap')
            bScale = np.pad(TPE/dz,((size,size),(size,size),(0,0),(0,0)),mode='wrap')
            ti3Scale=np.sqrt(TKE*TKE_v)
            tijScale=np.array([TKE,TKE,ti3Scale,TKE,ti3Scale,TKE_v])
        
        elif scalingStr=='global':
            Ug=fileUgs[ifile]
            Re=fileRes[ifile]
            b0=fileB0s[ifile]
            #g=9.8
            hvelScale = Ug/np.sqrt(Re)*np.ones(u.shape)
            vvelScale = Ug/np.sqrt(Re)*np.ones(u.shape)
            #bScale = g*np.ones(u.shape)
            bScale = b0*np.ones(u.shape)
            tijScale=Ug**2/Re*np.ones(outputFields.shape)

        elif scalingStr=='unscaled':
            reshape=False
            hvelScale = np.ones(u.shape)
            vvelScale = np.ones(u.shape)
            bScale = np.ones(u.shape)
            tijScale=np.ones(outputFields.shape)

        nx=outputFields.shape[1]
        ny=outputFields.shape[2]
        nz=outputFields.shape[3]
        nt=outputFields.shape[4]
        if maskdict==None:
            mask = np.random.rand(nx,ny,nz,nt) < filemaskpercents[ifile]
            maskDict["mask_"+file+'_'+str(irun)]=mask
        else:
            mask = maskdict["mask_"+file+'_'+str(irun)]
            
        krotList=list()
        for i in range(size, inputFields.shape[1] - size):
            for j in range(size, inputFields.shape[2] - size):
                for k in range(vsize, inputFields.shape[3] - vsize):
                    for it in range(inputFields.shape[4]):
                        if mask[i-size,j-size,k,it]:
                            ikrot=int(len(krotAll)*np.random.rand(1))
                            krot=krotAll[ikrot]
                            sample,label=myrotate_sample(inputFields[:,i - size: i + size + 1, j - size: j + size + 1, k - size: k + size + 1, it],outputFields[:,i-size,j-size,k,it],krot)
                            scaledInput=[scale(sample[0],sd=hvelScale[i, j, k, it]),
                                         scale(sample[1],sd=hvelScale[i, j, k, it]),
                                         scale(sample[2],sd=vvelScale[i, j, k, it]),
                                         scale(sample[3],sd=bScale[i, j, k, it])]
                            xList.append(scaledInput) 
                            yList.append(label)
                            tauScaleList.append(tijScale[:,i-size,j-size,k,it])
                            krotList.append(krot)
        maskDict["krot_"+file+'_'+str(irun)]=krotList
    y=np.array(yList)
    del yList
    x3d=np.array(xList)
    del xList
    tauScale=np.array(tauScaleList)
    del tauScaleList
    gc.collect()

    print("output shape is "+str(y.shape))
    print("input shape should be "+str(x3d.shape))
    if reshape:
        size=int(2*size+1)
        x=myreshape(x3d,size=size)                    
        print("input shape to do 3rd dimension as channel in R2Conv is "+str(x.shape))
    else:
        x=x3d
    
    return x, y, tauScale, maskDict

def myrotate_sample(inputFields_in,outputFields_in,krot):
    inputFields_out=np.empty(inputFields_in.shape)
    outputFields_out=np.empty(outputFields_in.shape)
    for v in range(inputFields_out.shape[0]):
        inputFields_out[v]=np.rot90(inputFields_in[v].copy(),krot) 
    theta=krot*np.pi/2.0
    
    R=np.rint([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) 
    for i in range(inputFields_out.shape[1]):
        for j in range(inputFields_out.shape[2]):
            for k in range(inputFields_out.shape[3]):
                inputFields_out[0:2,i,j,k]=R@inputFields_out[0:2,i,j,k] #u,v are index 0,1
    
    R=np.rint([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0,0,1]])
    T=np.array([[outputFields_in[0],outputFields_in[1], outputFields_in[2]],
                [outputFields_in[1],outputFields_in[3], outputFields_in[4]],
                [outputFields_in[2],outputFields_in[4], outputFields_in[5]]])
    Tprime=R@T@R.T
    outputFields_out[0]=Tprime[0,0]
    outputFields_out[1]=Tprime[0,1]
    outputFields_out[2]=Tprime[0,2]
    outputFields_out[3]=Tprime[1,1]
    outputFields_out[4]=Tprime[1,2]
    outputFields_out[5]=Tprime[2,2]

    return inputFields_out,outputFields_out

def preprocess_noB(files,filemaskpercents,scalingStr,fileUgs=None,fileRes=None,fileB0s=None,size=3,irun='', reshape=True):
    # Any of the below could be changed to inputs of the function
    path="/burg/glab/users/ac5006/DNStoLES/coarseData/fullTau/"
    #size = 1 # number of neighboring points to include in each input sample
    size=int((size-1)/2)
    vsize=1

    yList=list()
    xList=list()
    tauScaleList=list()
    maskDict = {}
    for ifile in range(len(files)): #tdqm notebook just draws progress bar
        file=files[ifile]
        ds=xr.open_dataset(path+file,decode_times=0)
        print(ds)

        # Getting input variables, removing laminar layers, wrapping periodic varaibles as reordering to x,y,z,t 
#         b=np.transpose(np.pad(cut_laminar(ds['b'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        u=np.transpose(np.pad(cut_laminar(ds['u'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap') , [2, 1, 0,3])
        v=np.transpose(np.pad(cut_laminar(ds['v'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        w=np.transpose(np.pad(cut_laminar(ds['w'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        b=np.random.normal(size=u.shape)
        inputFields = np.array([u, v, w, b])
        
        # Getting output fields
        tau_12 = np.transpose(cut_laminar(ds['tau12'].values), [2, 1, 0,3])
        tau_13 = np.transpose(cut_laminar(ds['tau13'].values), [2, 1, 0,3])
        tau_23 = np.transpose(cut_laminar(ds['tau23'].values), [2, 1, 0,3])
        tau_11 = np.transpose(cut_laminar(ds['tau11'].values), [2, 1, 0,3])
        tau_22 = np.transpose(cut_laminar(ds['tau22'].values), [2, 1, 0,3])
        tau_33 = np.transpose(cut_laminar(ds['tau33'].values), [2, 1, 0,3])
        outputFields = np.array([tau_11,tau_12, tau_13, tau_22, tau_23, tau_33])

        if scalingStr=='local':
            # Getting grid vairables to comput grid size
            x=ds['x'].values 
            yy=ds['y'].values
            z=cut_laminar(ds['z'].values)
            dx=np.mean(np.diff(x))
            dy=np.mean(np.diff(yy))
            dz=np.mean(np.diff(z))
            #delta=(dx*dy*dz)**(1.0/3.0)

            # Computing gradients for scaling saling variables: TKE, TPE, and N2
            d11=(u[2:,1:-1,:,:]-u[:-2,1:-1,:,:])/(2*dx)
            d12=(u[1:-1,2:,:,:]-u[1:-1,:-2,:,:])/(2*dy)
            d13=np.gradient(u[1:-1,1:-1,:,:],dz,axis=2)

            d21=(v[2:,1:-1,:,:]-v[:-2,1:-1,:,:])/(2*dx)
            d22=(v[1:-1,2:,:,:]-v[1:-1,:-2,:,:])/(2*dy)
            d23=np.gradient(v[1:-1,1:-1,:,:],dz,axis=2)

            d31=(w[2:,1:-1,:,:]-w[:-2,1:-1,:,:])/(2*dx)
            d32=(w[1:-1,2:,:,:]-w[1:-1,:-2,:,:])/(2*dy)
            d33=np.gradient(w[1:-1,1:-1,:,:],dz,axis=2)

            #dijdij = (d11*d11 + d12*d12 + d13*d13 + d21*d21 + d22*d22 + d23*d23 + d31*d31 + d32*d32 + d33*d33)
            
            TKE_h = dx**2*(d11**2 + d21**2) + dy**2*(d12**2 + d22**2) + dz**2*(d13**2 + d23**2) 
            TKE_v = dx**2*d31**2 + dy**2*d32**2 + dz**2*d33**2
            TKE = TKE_h + TKE_v
            #TKE = (delta**2/1.0)*dijdij
            del d11,d12,d13,d21,d22,d23,d31,d32,d33,TKE_h

#             db1=(b[2:,1:-1,:,:]-b[:-2,1:-1,:,:])/(2*dx)
#             db2=(b[1:-1,2:,:,:]-b[1:-1,:-2,:,:])/(2*dy)
#             db3=np.gradient(b[1:-1,1:-1,:,:],dz,axis=2)
#             N2=db3
#             #dbkdbk=db1*db1+db2*db2+db3*db3
#             TPE=(dx**2*db1**2+dy**2*db2**2+dz**2*db3**2)/N2
#             #TPE = (delta**2/1.0)*dbkdbk/N2
#             del db1,db2,db3 
            gc.collect()

            hvelScale = np.pad(np.sqrt(TKE),((size,size),(size,size),(0,0),(0,0)),mode='wrap')
            vvelScale = np.pad(np.sqrt(TKE_v),((size,size),(size,size),(0,0),(0,0)),mode='wrap')
#             bScale = np.pad(TPE/dz,((size,size),(size,size),(0,0),(0,0)),mode='wrap')
            ti3Scale=np.sqrt(TKE*TKE_v)
            tijScale=np.array([TKE,TKE,ti3Scale,TKE,ti3Scale,TKE_v])
        
        elif scalingStr=='global':
            Ug=fileUgs[ifile]
            Re=fileRes[ifile]
            b0=fileB0s[ifile]
            #g=9.8
            hvelScale = Ug/np.sqrt(Re)*np.ones(u.shape)
            vvelScale = Ug/np.sqrt(Re)*np.ones(u.shape)
            #bScale = g*np.ones(u.shape)
            bScale = b0*np.ones(u.shape)
            tijScale=Ug**2/Re*np.ones(outputFields.shape)

        elif scalingStr=='unscaled':
            reshape=False
            hvelScale = np.ones(u.shape)
            vvelScale = np.ones(u.shape)
            bScale = np.ones(u.shape)
            tijScale=np.ones(outputFields.shape)

        nx=outputFields.shape[1]
        ny=outputFields.shape[2]
        nz=outputFields.shape[3]
        nt=outputFields.shape[4]
        mask = np.random.rand(nx,ny,nz,nt) < filemaskpercents[ifile]
        maskDict["mask_"+file+'_'+str(irun)]=mask
        
        for i in range(size, inputFields.shape[1] - size):
            for j in range(size, inputFields.shape[2] - size):
                for k in range(vsize, inputFields.shape[3] - vsize):
                    for it in range(inputFields.shape[4]):
                        if mask[i-size,j-size,k,it]:
                            scaledInput=[scale(inputFields[0,i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it],sd=hvelScale[i, j, k, it]),
                                            scale(inputFields[1,i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it],sd=hvelScale[i, j, k, it]),
                                            scale(inputFields[2,i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it],sd=vvelScale[i, j, k, it]),
                                            scale(inputFields[3,i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it],sd=1)]#bScale[i, j, k, it])]
                            xList.append(scaledInput) 
                            yList.append(outputFields[:,i-size,j-size,k,it])
                            tauScaleList.append(tijScale[:,i-size,j-size,k,it])
                                 
    y=np.array(yList)
    del yList
    x3d=np.array(xList)
    del xList
    tauScale=np.array(tauScaleList)
    del tauScaleList
    gc.collect()

    print("output shape is "+str(y.shape))
    print("input shape should be "+str(x3d.shape))
    if reshape:
        size=int(2*size+1)
        x=myreshape(x3d,size=size)                    
        print("input shape to do 3rd dimension as channel in R2Conv is "+str(x.shape))
    else:
        x=x3d
        
    return x, y, tauScale, maskDict

def scale(x,mean=None,sd=None):

    if mean==None:
        mean=np.mean(x)
        
    if sd==None:
        sd=np.std(x)
    
    return (x - mean) / sd#, mean, sd

def cut_laminar(x):
    # Cuts the input 75% along the z-axis to remove laminar layers
    return x[0:int(0.75*x.shape[0])]

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