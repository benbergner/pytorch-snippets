import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Create a dummy model class with some parameters
class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, _):
        pass


if __name__ == "__main__":
    model = DummyModel()
    print(count_parameters(model))  # Output: 1059306
