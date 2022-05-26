import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
#import matplotlib.pyplot as plt
from src.model import *
from src.argparser import *
from src.utils import *
from src.dataloader import *
import pdb
import time
import os

def train(epoch):

    model.train()
    train_losses = torch.zeros(3)
    st1 = time.time()
    for batch_idx, (data) in enumerate(train_loader):
        #print(data.shape) # 64, 256, 128
        data = data.unsqueeze(1)
        #print(data.shape) # 64, 1, 256, 128
        data = data.to(device).view(-1,dimx)
        #print(data.shape) # 64, 32768
        optimizer.zero_grad()
        recon_y, mu_z, logvar_z, _ = model(data)
        loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)
        loss.backward()
        optimizer.step()
        train_losses[0] += loss.item()
        train_losses[1] += ELL.item()
        train_losses[2] += KLD.item()
        if batch_idx % args.log_interval == 0:
            en1 = time.time()
            print('Train Epoch:{} [{}/{} ({:.0f}%)] -ELL: {:5.2f} KLD: {:5.2f} Loss: {:5.2f} Time : {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                ELL.item() / len(data),KLD.item() / len(data),loss.item() / len(data), en1-st1))
            st1 = time.time()

    train_losses /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_losses[0]))

    return train_losses


def test(epoch, result_path):

    model.eval()
    test_losses = torch.zeros(3)

    with torch.no_grad():
        for batch_idx, (data_name, start, data) in enumerate(test_loader):
            #if batch_idx == 1:
            #    np.save('results/' + str(epoch) + '_GT_source_a.npy',source_2[0:3].cpu().numpy())
            #    np.save('results/' + str(epoch) + '_GT_source_b.npy',source_1[0:3].cpu().numpy())
            #    np.save('results/' + str(epoch) + '_GT_mixture.npy',data[0:3].cpu().numpy())
            #print(data.shape)
            print(batch_idx)
            data = data.unsqueeze(1)
            #print(data.shape)
            data = data.to(device).view(-1,dimx)
            #print(data.shape)

            recon_y, mu_z, logvar_z, recons = model(data)
            loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)

            test_losses[0] += loss.item()
            test_losses[1] += ELL.item()
            test_losses[2] += KLD.item()
        #print("True data:",data.size(0))
        n = min(data.size(0), 6)
        ncols = (2+args.sources)
        comparison = torch.zeros(n*ncols,1,data_size,data_len)
        print(comparison.shape, data.shape) #24, 256, 32, 53, 8192
        comparison[::ncols] = data.view(data.size(0), 1, data_size, data_len)[:n]
        print(comparison[::ncols].shape)
        comparison[1::ncols] = recon_y.view(data.size(0), 1, data_size, data_len)[:n]
        print(comparison[1::ncols].shape)
        for i in range(args.sources):
            comparison[(i+2)::ncols] = recons[:,i].view(data.size(0), 1, data_size, data_len)[:n] #::는 거꾸로?
        grid = make_grid(comparison,nrow=ncols)
        s1_p = data_name[0] + "/bass.npy"
        s2_p = data_name[0] + "/drums.npy"
        s3_p = data_name[0] + "/other.npy"
        s4_p = data_name[0] + "/vocals.npy"
        source_1 = np.load(s1_p)[:,start[0]:start[0]+data_len]
        source_2 = np.load(s2_p)[:,start[0]:start[0]+data_len]
        source_3 = np.load(s3_p)[:,start[0]:start[0]+data_len]
        source_4 = np.load(s4_p)[:,start[0]:start[0]+data_len]
        if source_1.ndim ==3:
            source_1_real = source_1[:,:,0] + source_1[:,:,1] * 1j
            source_1 = np.absolute(source_1_real)
        else:
            source_1 = np.absolute(source_1)
        if source_2.ndim ==3:
            source_2_real = source_2[:,:,0] + source_2[:,:,1] * 1j
            source_2 = np.absolute(source_2_real)
        else:
            source_2 = np.absolute(source_2)
            
        if source_3.ndim ==3:
            source_3_real = source_3[:,:,0] + source_3[:,:,1] * 1j
            source_3 = np.absolute(source_3_real)
        else:
            source_3 = np.absolute(source_3)
        if source_4.ndim ==3:
            source_4_real = source_4[:,:,0] + source_4[:,:,1] * 1j
            source_4 = np.absolute(source_4_real)
        else:
            source_4 = np.absolute(source_4)

                
        source = np.zeros((4,source_1.shape[0],source_1.shape[1]))
        
        source[0] = source_1
        source[1] = source_2
        source[2] = source_3
        source[3] = source_4

        #save_image(comparison.cpu(),result_path + str(epoch) + '.png', nrow=ncols)
        
        np.save(result_path + str(epoch) + '.npy',comparison.numpy())
        np.save(result_path + str(epoch) + '_GT_mixture.npy',data[0:3].cpu().numpy())
        np.save(result_path + str(epoch) + '_sources.npy',source)
        test_losses /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_losses[0]))

    return test_losses
'''
def plot_losses(losses):
	plt.figure()
	plt.plot(np.array(range(1,args.epochs+1)),losses["train"][:,0].view(-1),label="Train")
	plt.plot(np.array(range(1,args.epochs+1)),losses["test"][:,0].view(-1),label="Test")
	plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.xlim(1,args.epochs)
	plt.savefig('results/losses.png')
	plt.close()
'''


args = parser.parse_args()
torch.manual_seed(args.seed)
data_size = args.data_size
data_len = args.data_len
result_path = args.result_path
print(result_path)
os.makedirs(result_path, exist_ok=True)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
print("Batchsize", args.batch_size, "data_directory :",args.data_directory)
train_loader, test_loader = get_data_loaders(args.data_directory,args.batch_size,data_size,data_len,kwargs)

# MNIST is 28 X 28

#dimx = int(28*28)
dimx = int(data_size * data_len)

model = VAE(dimx=dimx,dimz=args.dimz,n_sources=args.sources,device=device,variational=args.variational).to(device)
loss_function = Loss(sources=args.sources,likelihood='laplace',variational=args.variational,prior=args.prior,scale=args.scale)

optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay, last_epoch=-1)

losses = {"train": torch.zeros(args.epochs,3), "test": torch.zeros(args.epochs,3)}

for epoch in range(1, args.epochs+1):
    beta = min(1.0,(epoch)/min(args.epochs,args.warm_up)) * args.beta_max

    losses["train"][epoch-1] = train(epoch)
    losses["test"][epoch-1] = test(epoch, result_path)

    if optimizer.param_groups[0]['lr'] >= 1.0e-5:
        scheduler.step()

    with torch.no_grad():
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),result_path+('vae' if args.variational else 'ae')+'_K' + str(args.sources) +  '.pt')


#plot_losses(losses)
