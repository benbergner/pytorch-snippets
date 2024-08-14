import torch
import torch.nn as nn

"""
Gradient accumulation is a technique to train models with larger batch sizes than the GPU memory can handle.
Instead of updating the model parameters every batch, the gradients are accumulated over multiple batches.
After a certain number of batches, the accumulated gradients are used to update the model parameters.

In this snippet, two functions are provided to perform gradient accumulation:
- grad_accum_simple: a simple implementation of gradient accumulation
- grad_accum_amp: an implementation of gradient accumulation with automatic mixed precision (AMP)

The functions take the following arguments:
- data_loader: the data loader
- model: the model
- optimizer: the optimizer
- loss_fn: the loss function
- fp16_scaler: the AMP scaler (only for grad_accum_amp)
- accum_steps: the number of gradient accumulation steps

The functions iterate over the data loader and perform the following steps:
1. Load the input and target data
2. Forward pass
3. Compute the loss
4. Backward pass
5. Update the model parameters every accum_steps steps

The grad_accum_amp function uses the torch.cuda.amp.autocast context manager to enable automatic mixed precision.
The fp16_scaler argument is used to scale the loss value before calling the backward method.
The scaler.step method is called to update the model parameters, and the scaler.update method is called to update the scaler.

Example usage:
grad_accum_simple(data_loader, model, optimizer, loss_fn, accum_steps)
grad_accum_amp(data_loader, model, optimizer, loss_fn, fp16_scaler, accum_steps)

References:
- https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
"""


def grad_accum_simple(data_loader, model, optimizer, loss_fn, accum_steps):
    for i, (input, target) in enumerate(data_loader):

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        # Compute the loss and divide by the number of accumulation steps
        loss = loss_fn(output, target) / accum_steps

        # Backward pass: compute and add/accumulate the gradients
        loss.backward()

        # Update the model parameters every accum_steps steps
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


def grad_accum_amp(data_loader, model, optimizer, loss_fn, fp16_scaler, accum_steps):
    for i, (input, target) in enumerate(data_loader):

        input = input.cuda()
        target = target.cuda()

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=fp16_scaler is not None
        ):
            output = model(input)
            loss = loss_fn(output, target) / accum_steps

        if fp16_scaler is not None:
            fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if fp16_scaler is not None:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()


####################################################################################################
# Example usage
####################################################################################################


# Dummy model for testing
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


# Dummy dataset for testing
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [torch.randn(3) for _ in range(4)]
        self.targets = [torch.randn(1) for _ in range(4)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


if __name__ == "__main__":

    dataset = DummyDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = DummyModel().cuda()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fp16_scaler = torch.cuda.amp.GradScaler()
    accum_steps = 2

    grad_accum_simple(data_loader, model, optimizer, loss_fn, accum_steps)
    grad_accum_amp(data_loader, model, optimizer, loss_fn, fp16_scaler, accum_steps)
