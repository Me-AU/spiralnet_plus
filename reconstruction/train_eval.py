import time
import os
import torch
import torch.nn.functional as F


def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, device, beta=1.0):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, train_loader, device, beta)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device, beta)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(model, optimizer, loader, device, beta=1.0):
    model.train()

    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        x = data.x.to(device)

        # Check if model is AE or VAE based on the number of outputs
        output = model(x)
        if isinstance(output, tuple) and len(output) == 3:
            recon, mu, logvar = output  # VAE
            recon_loss = F.l1_loss(recon, x, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
        else:
            recon = output  # AE
            loss = F.l1_loss(recon, x, reduction='mean')

        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader)


def test(model, loader, device, beta=1.0):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for data in loader:
            x = data.x.to(device)

            output = model(x)
            if isinstance(output, tuple) and len(output) == 3:
                recon, mu, logvar = output  # VAE
                recon_loss = F.l1_loss(recon, x, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss
            else:
                recon = output  # AE
                loss = F.l1_loss(recon, x, reduction='mean')

            total_loss += loss.item()

    return total_loss / len(loader)


def eval_error(model, test_loader, device, meshdata, out_dir):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for data in test_loader:
            x = data.x.to(device)
            pred = model(x)
            if isinstance(pred, tuple):  # Handle VAE output
                pred = pred[0]  # Only take the reconstruction
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_x *= 1000

            tmp_error = torch.sqrt(torch.sum((reshaped_pred - reshaped_x) ** 2, dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]
        mean_error = new_errors.view((-1,)).mean()
        std_error = new_errors.view((-1,)).std()
        median_error = new_errors.view((-1,)).median()

    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error, median_error)
    out_error_fp = os.path.join(out_dir, 'euc_errors.txt')
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)
