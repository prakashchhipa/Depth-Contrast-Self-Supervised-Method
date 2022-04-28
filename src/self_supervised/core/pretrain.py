import os,sys
import logging
from os import get_exec_path
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from tqdm import tqdm

import torch
from self_supervised.core import ssl_loss
from self_supervised.apply import config
sys.path.append(os.path.dirname(__file__))


def pretrain_epoch_simCLR(gpu, current_epoch, epochs, batch_size, train_loader,
                          model, optimizer, criterion):

    model.train()
    total_loss = 0
    epoch_response_dir = {}
    with tqdm(total=batch_size * len(train_loader),
              desc=f'Epoch {current_epoch}/{epochs}',
              unit='img') as (pbar):

        for idx, batch in enumerate(train_loader):
            view1, view2 = batch[0], batch[1]
            
            #print(type(view1), view1)
            
            
            #b, c, h, w = view1.size()
            #for A transform
            #view1 = view1['image'].cuda(gpu, non_blocking=True)
            #view2 = view2['image'].cuda(gpu, non_blocking=True)
            #for pytorch tranform
            view1 = view1.cuda(gpu, non_blocking=True)
            view2 = view2.cuda(gpu, non_blocking=True)

            output_view1 = model(view1)
            output_view2 = model(view2)

            output = torch.cat(
                [output_view1.unsqueeze(1),
                 output_view2.unsqueeze(1)], dim=1)
            loss = criterion(output)
            curr_loss = loss.item()
            total_loss += curr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''logging'''
            #logging.info('minibatch: {idx} simCLR running_loss: {loss.item()}')
            (pbar.set_postfix)(**{'loss (batch)': loss.item()})
            pbar.update(view1.shape[0])
            #print(file_path)

        # Prepare epoch reponse and return
        epoch_response_dir['model'] = model
        epoch_response_dir['loss'] = total_loss/(batch_size*len(train_loader))
        epoch_response_dir['image_pair'] = [view1, view2]

    return epoch_response_dir


def pretrain_epoch_simsiam(gpu, current_epoch,
                           epochs,
                           batch_size,
                           train_loader,
                           model,
                           optimizer,
                           criterion=ssl_loss.simsiam_loss):
    model.train()
    total_loss = 0
    epoch_response_dir = {}
    with tqdm(total=batch_size * len(train_loader),
              desc=f'Epoch {current_epoch}/{epochs}',
              unit='img') as pbar:
        for i, batch in enumerate(train_loader):
            view1, view2 = batch
            b, c, h, w = view1.size()
            x1 = view1.cuda(gpu, non_blocking=True)
            x2 = view2.cuda(gpu, non_blocking=True)

            z1, z2 = model.backbone_mlp(x1), model.backbone_mlp(x2)
            p1, p2 = model.head(z1), model.head(z2)

            loss = (criterion(p1, z2) / 2) + (criterion(p2, z1) / 2)
            curr_loss = loss.item()
            total_loss += curr_loss/batch_size
            #print(f'minibatch: {i} running_loss: {loss.item()}', end='\r')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''logging'''
            '''logging.info(
                'minibatch: {idx} simsiam running_loss: {loss.item()}',
                end='\r')'''
            (pbar.set_postfix)(**{'loss (batch)': loss.item()})
            pbar.update(view1.shape[0])

            # Prepare epoch reponse and return
            epoch_response_dir['model'] = model
            epoch_response_dir['loss'] = total_loss/(len(train_loader))
            epoch_response_dir['image_pair'] = [view1, view2]

    return epoch_response_dir


def pretrain_BYOL(current_epoch,
                  epochs,
                  batch_size,
                  train_loader,
                  model,
                  optimizer,
                  criterion=None):
    model.train()
    total_loss = 0
    epoch_response_dir = {}
    with tqdm(total=batch_size * len(train_loader),
              desc=f'Epoch {current_epoch}/{epochs}',
              unit='img') as pbar:
        for i, images in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            loss = model(images)
            curr_loss = loss.item()
            total_loss += curr_loss
            print(f'minibatch: {i} running_loss: {loss.item()}', end='\r')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''logging'''
            logging.info('minibatch: {idx} BYOL running_loss: {loss.item()}',
                         end='\r')
            (pbar.set_postfix)(**{'loss (batch)': loss.item()})
            pbar.update(images.shape[0])

            # Prepare epoch reponse and return
            epoch_response_dir['model'] = model
            epoch_response_dir['loss'] = total_loss
            epoch_response_dir['image_pair'] = [images, images]

    return model, total_loss