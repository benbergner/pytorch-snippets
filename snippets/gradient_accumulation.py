import torch

"""
Gradient accumulation is a technique to train models with larger batch sizes than the GPU memory can handle.
Instead of updating the model parameters every batch, the gradients are accumulated over multiple batches.
After a certain number of batches, the accumulated gradients are used to update the model parameters.

In this snippet, we provide two functions to perform gradient accumulation:
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

data_loader = ...  # your data loader
model = ...  # your model
optimizer = ...  # your optimizer
loss_fn = ...  # your loss function
fp16_scaler = torch.cuda.amp.GradScaler()  # AMP scaler

accum_steps = 2  # Number of gradient accumulation steps


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

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            output = model(input)
            loss = loss_fn(output, target) / accum_steps

        if fp16_scaler is not None:
            fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

        loss.backward()

        if (i + 1) % accum_steps == 0:
            if fp16_scaler is not None:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
