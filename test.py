# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import os
# import torch
# def test(model, lr_image_path, save_path='output.png'):
#     model.eval()  # Set model to evaluation mode
#     image = Image.open(lr_image_path)
#     transform = transforms.ToTensor()
#     lr_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
#
#     # Forward pass to get super-resolved image
#     with torch.no_grad():
#         sr_image = model(lr_image)
#
#     # Convert tensor to image format
#     sr_image = sr_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Remove batch and channel dimension
#     sr_image = (sr_image * 255.0).astype(np.uint8)
#
#     # Save or display the image
#     plt.imshow(sr_image)
#     plt.axis('off')
#     plt.savefig(save_path)