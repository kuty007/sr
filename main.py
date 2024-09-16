import train, data_load

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

def main():
    # Define image transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(2040, 1356)),
        transforms.ToTensor(),
    ])

    # Load dataset
    train_dataset = data_load.SRDataset('data/lr_x2', 'data/gt', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Instantiate the model, loss function, and optimizer
    model = train.EDSR().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    print("Training the model...")
    train.train(model, train_loader, criterion, optimizer)

    # Save the model
    torch.save(model.state_dict(), 'edsr_super_resolution.pth')
    print("Model saved!")

    # Test the model on a low-resolution image
    print("Testing the model...")
    # test(model, 'path/to/test_lr_image.png', 'output_sr_image.png')
    # print("Test complete! Super-resolved image saved as 'output_sr_image.png'")


if __name__ == '__main__':
    main()
