import torch.optim as optim
from torch import nn
import torch

# from data_load import train_loader
from net import EDSR

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model, loss function, and optimizer
model = EDSR().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for lr_images, hr_images in train_loader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            sr_images = model(lr_images)

            # Compute loss
            loss = criterion(sr_images, hr_images)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# Start training
# train(model, train_loader, criterion, optimizer)
# Save the trained model

