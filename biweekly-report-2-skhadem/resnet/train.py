from resnet import ResNet, BottleNeckBlock, BasicBlock
from data_loaders import get_data_loaders

import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block_type=BasicBlock, depths=[2, 2, 2, 2])

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block_type=BasicBlock, depths=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block_type=BottleNeckBlock, depths=[3, 4, 6, 3])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--depth', type=int, default=50)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # TODO: Add these as args
    batch_size = 128
    base_lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay = 0.1
    log_freq = 150

    if args.depth not in [18, 34, 50]:
        raise NotImplementedError
    
    if args.depth == 18:
        model = resnet18(3, 10)
        run_name = 'resnet-18'
    if args.depth == 34:
        model = resnet34(3, 10)
        run_name = 'resnet-34'
    if args.depth == 50:
        model = resnet50(3, 10)
        run_name = 'resnet-50'

    model.cuda()

    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20],
        gamma=lr_decay)

    loss_fcn = torch.nn.CrossEntropyLoss(size_average=True)

    writer = SummaryWriter('./runs/%s'%run_name)
    batch_count = 0
    for epoch in range(args.epochs):
        scheduler.step()

        for step, (data, targets) in enumerate(train_loader):
            batch_count += 1
            # Move data to gpu
            data = data.cuda()
            targets = targets.cuda()

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)
            # Backprop
            loss = loss_fcn(outputs, targets)
            loss.backward()
            
            # Step optimizer
            optimizer.step()

            if step % log_freq == 0:
                writer.add_scalar('training_loss', loss.item(), global_step=batch_count)

                print('%s/%s | Loss: %0.4f'%(step, len(train_loader), loss.item()))

        print('-'*20)
        print('Done with epoch: %s'%epoch)

    # Save model
    torch.save(model.state_dict(), './weights/%s.pt'%run_name)

if __name__ == '__main__':
    main()