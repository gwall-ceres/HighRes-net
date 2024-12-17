from torchvision.models import vgg19
import torch



# Define the VGGFeatureExtractor with updated weights parameter
class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, layers=['0', '5', '10', '19', '28']):
        super(VGGFeatureExtractor, self).__init__()
        weights_path="./vgg19_conv_layers.pth"
        self.layers = layers
        self.vgg = vgg19().features[:29]  # Match truncated layers
        # Load weights from the saved file
        self.vgg.load_state_dict(torch.load(weights_path, map_location="cpu"))

        if torch.cuda.is_available():
            self.hardware = 'cuda'
            print("VGG using CUDA")
        else:
            self.hardware = 'cpu'

    def forward(self, x):
        outputs = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                outputs[name] = x
        return outputs

    @staticmethod
    def init_VGG_for_perceptual_loss():
        # Initialize the feature extractor
        feature_extractor = VGGFeatureExtractor(layers=['0', '5', '10', '19', '28']).to('cuda' if torch.cuda.is_available() else 'cpu')
        feature_extractor.eval()
        # Disable gradient computations
        for param in feature_extractor.parameters():
            param.requires_grad = False

        return feature_extractor