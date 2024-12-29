import torch
from tqdm import tqdm      # pretty progress bar

def train_epoch(data_loader, model, optimiser, criterion, device):

  # set model to training mode. This is important because some layers behave differently during training and testing
  model.train(True)
  model.to(device)

  # stats
  loss_total = 0.0
  oa_total = 0.0

  # iterate over dataset
  pBar = tqdm(data_loader)
  #for idx, (data, target) in enumerate(tqdm(data_loader)):
  for idx, (data, target) in enumerate(pBar):
    # put data and target onto correct device
    data, target = data.to(device), target.to(device)
    # ! change to dataloader or dataset
    data = data.to(torch.float32)   # to match weights of model
    target = target.to(torch.float32) # to match data of model

    #maybe this is the solution
    #data.requires_grad = True
    # reset gradients
    optimiser.zero_grad()

    # forward pass
    pred = model(data)

    # loss
    target = target.squeeze()  # to match shape of pred
    loss = criterion(pred, target)

    # backward pass
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # parameter update
    optimiser.step()

    # stats update
    loss_total += loss.item()
    # ! probably need to change
    acc = torch.mean(torch.abs(torch.sub(pred, target))).item()
    oa_total += acc
    # print('loss : ', loss.item())
    # format progress bar
    pBar.set_description('Loss: {:.2f}, OA: {:.2f}'.format(
      loss_total/(idx+1),
      oa_total/(idx+1)
    ))
    pBar.update(1)
  
  pBar.close()

  # normalise stats
  loss_total /= len(data_loader)
  oa_total /= len(data_loader)

  return model, loss_total, oa_total


# ------------------------------------------------------------------------------------
def validate_epoch(data_loader, model, device):       # note: no optimiser needed

  # set model to evaluation mode
  model.train(False)
  model.to(device)

  # stats
  loss_total = 0.0
  oa_total = 0.0

  # iterate over dataset
  pBar = trange(len(data_loader))
  for idx, (data, target) in enumerate(data_loader):
    with torch.no_grad():

      #TODO: likewise, implement the validation routine. This is very similar, but not identical, to the training steps.

      # put data and target onto correct device
      data, target = data.to(device), target.to(device)
      # ! change to dataloader or dataset
      data = data.to(torch.float32)   # to match weights of model
      target = target.to(torch.float32) # to match data of model

      # forward pass
      pred = model(data)

      # loss
      loss = criterion(pred, target)

      # stats update
      loss_total += loss.item()
      acc = torch.mean(torch.abs(torch.sub(pred, target))).item()
      oa_total += acc

      # format progress bar
      pBar.set_description('Loss: {:.2f}, OA: {:.2f}'.format(
        loss_total/(idx+1),
        100 * oa_total/(idx+1)
      ))
      pBar.update(1)

  pBar.close()

  # normalise stats
  loss_total /= len(data_loader)
  oa_total /= len(data_loader)

  return loss_total, oa_total