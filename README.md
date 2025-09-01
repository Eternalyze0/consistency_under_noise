# consistency_under_noise
Noise is added to the input and then a consistency loss is applied to the noised and noiseless outputs.

Baseline:
```
Test set: Average loss: 0.0259, Accuracy: 9908/10000 (99%)
```
Consistency Under Noise:
```
Test set: Average loss: 0.0239, Accuracy: 9922/10000 (99%)
```
Code:
```py
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output_noise = model(data + torch.randn_like(data))
        loss = F.nll_loss(output, target) + 0.001 * F.mse_loss(output_noise, output)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
```
