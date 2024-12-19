from torchvision.models import vgg19
from torchvision import transforms
import torch
import numpy as np


 #[0.485, 0.456, 0.406],  # ImageNet means
 #[0.229, 0.224, 0.225]
# Define the VGGFeatureExtractor with updated weights parameter
class VGGFeatureExtractor(torch.nn.Module):
    #layers=['0', '5', '10', '19', '28']
    layers=['1', '6', '11', '20', '29']

    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()

        if torch.cuda.is_available():
            self.hardware = 'cuda'
            print("VGG using CUDA")
        else:
            self.hardware = 'cpu'

        #from torchvision.models import VGG19_Weights
        #weights = VGG19_Weights.DEFAULT
        weights_path="./vgg19_conv_layers.pth"
        #self.layers = layers
        self.vgg = vgg19().features[:30]  # Match truncated layers
        #for idx, layer in enumerate(self.vgg):
        #    print(idx, layer)
        # Load weights from the saved file
        self.vgg.load_state_dict(torch.load(weights_path, map_location=self.hardware))
        self.preprocess = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        #self.vgg = vgg19(weights=weights).features[:int(VGGFeatureExtractor.layers[-1])+1]
        

        

    def forward(self, x):
        #x = self.convert_grayscale_to_input_tensor(x).to(self.hardware)
        outputs = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in VGGFeatureExtractor.layers:
                outputs[name] = x
        return outputs
    

    def convert_grayscale_to_input_tensor(self, x):
        '''
        Convert a grayscale image to a 4-channel input tensor: [1, 3, H, W]

        '''
        if x.ndim == 2:
            # Grayscale image, replicate channels to make (H, W, 3)
            x = np.stack([x, x, x], axis=-1)
        elif x.ndim == 3 and x.shape[2] == 1:
            # Single-channel image with shape (H, W, 1), replicate to (H, W, 3)
            x = np.concatenate([x, x, x], axis=2)
        elif x.ndim == 3 and x.shape[2] == 3:
            pass
        else:
            raise ValueError("Input image must have shape (H, W), (H, W, 1), or (H, W, 3)")

        x = self.preprocess(x).unsqueeze(0) # Shape: [1, 3, H, W]

        return x

    @staticmethod
    def init_VGG_for_perceptual_loss():
        # Initialize the feature extractor
        feature_extractor = VGGFeatureExtractor().to('cuda' if torch.cuda.is_available() else 'cpu')
        feature_extractor.eval()
        # Disable gradient computations
        for param in feature_extractor.parameters():
            param.requires_grad = False

        return feature_extractor