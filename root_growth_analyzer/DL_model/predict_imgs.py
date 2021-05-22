# Read libraries
from os import getcwd
from pathlib import Path
from PIL import Image
import torch
import torchvision
from skimage.morphology import erosion
import matplotlib.pyplot as plt
import time
import logging
from torchvision.transforms import functional as F

# Import model
from root_growth_analyzer.DL_model.model import SegNet

# Set thershold
#threshold=0.9 # threshold prediction

# Define image folder paths
#read_dir = Path("data/prediction_data") # Directory to test data
#result_dir = "data/prediction_result" # Directory to test data

# Define paths
#weights_path = "trained_segnet_weights.pt" # Path to trained model weights


# Image normalization
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std)


class DLModel():

    threshold = 0.9 # threshold prediction
    weights_path = f'root_growth_analyzer/DL_model/trained_segnet_weights.pt' # Path to trained model weights

    # Pad width and size to be dividable by 16
    def pad_256(self, img_path):
        # Open image
        image = Image.open(img_path)
        # Resize image. 1 pixel = 0.1mm.
        image = torchvision.transforms.Resize([2100, 2970])(image)
        # Cut 50 pixels from each border. Still 1 pixel = 0.1mm.
        image  = torchvision.transforms.CenterCrop([2000, 2870])(image)
        # Image dimensions
        W, H = image.size
        new_w = ((W - 1) // 256 + 1) * 256
        new_h = ((H - 1) // 256 + 1) * 256
        new_img = Image.new("RGB", (new_w, new_h))
        new_img.paste(image, ((new_w - W) // 2, (new_h - H) // 2))
        NW, NH = new_img.size
        # To tensor
        new_img = torchvision.transforms.ToTensor()(new_img)
        # Normalize
        normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        new_img = normalize(new_img)
        return new_img, (H, W, NH, NW)

    # Make prediction
    def predict(self, model, test_img, device):
        # Desable gradient saving
        for p in model.parameters():
            p.requires_grad = False
        # Set model to eval mode
        model.eval()
        # Unsqueeze image. Test_img.shape = (3, 2304, 2560).
        test_img = test_img.unsqueeze(0)
        # make prediction
        output = model(test_img)
        # Squeeze back the image. Output.shape = (1, 1, 2304, 2560).
        output = torch.squeeze(output)
        # Empty cuda caches. May be useless.
        torch.cuda.empty_cache()
        return output

    # Prediction main function
    def predict_gen(self, model, img_path, thres, device, result_dir):
        # Create padded image
        img, dims = self.pad_256(img_path)
        # Get padded and original image sizes
        height, width, padded_height, padded_width = dims
        # padded image to device
        img = img.to(device)
        # Make prediction
        prediction = self.predict(model, img, device)
        # Prediction from tensor to numpy
        if device.type == "cpu":
            prediction = prediction.detach().numpy()
        else:
            prediction = prediction.cpu().detach().numpy()
        # Threshold prediction
        prediction[prediction >= thres] = 1.0
        prediction[prediction < thres] = 0.0

        # Remove padding
        prediction = prediction[
            (padded_height - height) // 2 : (padded_height - height) // 2 + height,
            (padded_width - width) // 2 : (padded_width - width) // 2 + width
        ]
        # Set prediction image name and path
        new_file_name = img_path.parts[-1].split(".jpg")[0] + "-prediction.jpg"
        save_dir = result_dir + "/" + new_file_name
        # Save prediction image
        plt.imsave(save_dir, prediction, cmap="gray")
        # Print saved image name and info that generation has completed
        logging.info("{} generated!".format(new_file_name))

    # The main
    def apply_dl_model(self, read_dir, result_dir, progress=None, prog_step=0, exit_flag=None):
        read_dir = Path(read_dir)
        # If available, use GPU, otherwise use CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Model to device
        model = SegNet(8, 5).to(device)
        # Load trained model weights
        if device.type == "cpu":
            logging.info("No Cuda available, will use CPU")
            model.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
        else:
            logging.info("Use Cuda GPU")
            model.load_state_dict(torch.load(self.weights_path))
        # Get image paths
        img_paths = read_dir.glob("*.jpg")

        # Loop image paths and make predictions
        for img_path in img_paths:
            if exit_flag and exit_flag.is_set():
                logging.info("Received exit signal, terminating run")
                return
            self.predict_gen(model, img_path, self.threshold, device, result_dir)
            if progress:
                progress.step(prog_step)
