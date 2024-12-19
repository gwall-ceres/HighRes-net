import torch
from torchvision.models import vgg19, VGG19_Weights

# Load VGG19 and keep only convolutional layers up to a specific layer
vgg_model = vgg19(weights=VGG19_Weights.DEFAULT).features[:30]  # Layers up to '28'

# Save the reduced model to disk
torch.save(vgg_model.state_dict(), "vgg19_conv_layers.pth")
print("Saved convolutional layers to vgg19_conv_layers.pth")

