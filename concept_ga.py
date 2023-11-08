import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam


# Define the dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Define the model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


# Define the training loop
def train(model, dataset, batch_size, num_steps, accumulation_steps):
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for step in range(num_steps):
        total_loss = 0

        # Accumulate gradients over multiple batches
        for i, batch in enumerate(dataloader):
            x, y = batch
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss /= accumulation_steps
            loss.backward()
            total_loss += loss.item()

            # Update gradients every accumulation_steps batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()

        # Print average loss for the epoch
        print(f"Step {step}: average loss = {total_loss / len(dataloader)}")


# Create some dummy data
data = [(torch.randn(10), torch.randn(1)) for _ in range(100)]

# Create the model and dataset
model = MyModel()
dataset = MyDataset(data)

# Train the model with gradient accumulation
train(model, dataset, batch_size=10, num_steps=10, accumulation_steps=5)
