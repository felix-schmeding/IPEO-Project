from src.data.CanopyDataset import CanopyDataset
from src.train.epoch import train_epoch, validate_epoch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch import nn

from src.models.SIDE_code_decode import SIDE

# define hyperparameters
device = 'cuda'
start_epoch = 0        # set to 0 to start from scratch again or to 'latest' to continue training from saved checkpoint
batch_size = 1
learning_rate = 0.001
weight_decay = 0.001
num_epochs = 1

def setup_optimiser(model, learning_rate, weight_decay):
  return SGD(
    model.parameters(),
    learning_rate,
  )

# we also create a function for the data loader here (see Section 2.6 in Exercise 6)
def load_dataloader(batch_size, dataset, split='train'):
  return DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=(split=='train'),       # we shuffle the image order for the training dataset
      num_workers=2                   # perform data loading with two CPU threads
  )

if __name__ == '__main__':
    # * create all the needed variables
    train_df = CanopyDataset(split='train')
    #val_df = CanopyDataset(split='validation')

    dl_train = load_dataloader(batch_size, train_df)
    #dl_val_test = load_dataloader(batch_size, val_df)

    # model
    model = SIDE()

    # optimizer
    optim = setup_optimiser(model, learning_rate, weight_decay)

    criterion = nn.MSELoss()

    _, loss_total, oa_total = train_epoch(data_loader=dl_train, model=model, optimiser=optim, criterion=criterion, device=device)
    print(f'Loss: {loss_total}, OA: {oa_total}')









# def do_epochs(dl_train_test, dl_val_test, model, optim_test, device, num_epochs, start_epoch=0):
#     # do epochs
#     while start_epoch < num_epochs:
#         # training
#         model, loss_train, oa_train = train_epoch(dl_train_test, model, optim_test, device)

#         # validation
#         loss_val, oa_val = validate_epoch(dl_val_test, model, device)

#         # print stats
#         print('[Ep. {}/{}] Loss train: {:.2f}, val: {:.2f}; OA train: {:.2f}, val: {:.2f}'.format(
#             start_epoch+1, num_epochs,
#             loss_train, loss_val,
#             100*oa_train, 100*oa_val
#         ))

#         # save model
#         start_epoch += 1
#         save_model(model, start_epoch)

