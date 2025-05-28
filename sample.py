from script import DDPM, ContextUnet, ddpm_schedules
import torch
from torchvision.utils import save_image, make_grid
import os
from phasePrediction2 import phasePredictionLite

def load_ddpm_model(model_path, device, n_classes=7, n_T=500, betas=(1e-4, 0.02)):
    """
    Load the DDPM model with pretrained weights.

    Args:
        model_path (str): Path to the saved model weights.
        device (str): Device to load the model on (e.g., 'cuda' or 'cpu').
        n_classes (int): Number of conditioning classes.
        n_T (int): Number of diffusion timesteps.
        betas (tuple): Beta schedule for the diffusion process.

    Returns:
        DDPM: The loaded DDPM model.
    """
    # Initialize the U-Net model
    unet = ContextUnet(in_ch=1, base_ch=128, cond_dim=n_classes)
    
    # Initialize the DDPM model
    ddpm = DDPM(unet, T=n_T, betas=betas, device=device)
    
    # Load the model weights
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.to(device)
    ddpm.eval()
    print(f"Model loaded from {model_path}")
    return ddpm

def sample_images(ddpm, n_samples, image_size, condition, guide_weight, save_dir, save_seq=False):
    """
    Generate and save sample images using the DDPM model.

    Args:
        ddpm (DDPM): The trained DDPM model.
        n_samples (int): Number of samples to generate.
        image_size (tuple): Size of the generated images (channels, height, width).
        condition (torch.Tensor): Conditioning vector for the samples.
        guide_weight (float): Weight for classifier-free guidance.
        save_dir (str): Directory to save the generated images.
    """
    ddpm.eval()
    with torch.no_grad():
        # Generate samples
        generated_images, generated_images_seq = ddpm.sample(n_samples, image_size, condition, guide_w=guide_weight)
        
        # Create a grid of generated images
        # grid = make_grid(generated_images * -1 + 1, nrow=10)
        if save_seq:
            print(f"Generated images shape: {generated_images.shape}")
            concatenated_images = torch.cat(generated_images_seq, dim=0)  # Shape: [num_images, 1, 128, 128]
            print(f"Concatenated images shape: {concatenated_images.shape}")
            grid = make_grid(concatenated_images, nrow=10)
        else:
            grid = make_grid(generated_images, nrow=10)
        # Save the grid as an image
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "imaginary.png")
        save_image(grid, save_path)
        print(f"Generated samples saved at {save_path}")

if __name__ == "__main__":
    # Configuration
    model_path = "./data/diffusion_outputs2/model_99.pth"  # Path to the saved model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples = 1 #10
    image_size = (1, 128, 128)
    n_classes = 7
    guide_weight = 0.5
    save_dir = "./generated_images"
    c_i = []
    modification = 0
    cooling_rate = 0.4
    composition = {"Si": 0.05, "Cu": 0.0125, "Mg": 0.005, "Al": 0.9325}  # Example composition
    phasePredictor = phasePredictionLite('./matQuery/database/AlSiMgCuScheil.pkl')
    save_seq = True
    c_i.append(modification)
    c_i.append(cooling_rate)
    res= phasePredictor.predict(composition)
    c_i.extend([res[phase] for phase in res.keys() if phase not in ["FCC_A1", "S_PHASE", "T_PHASE"]])
    print(f"Conditioning vector: {c_i}")
    assert len(c_i) == n_classes, f"Expected {n_classes} classes, got {len(c_i)}"
    c_i = torch.tensor(c_i, dtype=torch.float32).view(1, -1).to(device)
    # Example conditioning vector (random for demonstration)
    # c_i = torch.randn(n_samples, n_classes).to(device)
    # c_i = torch.tensor([1, 0.5, 0, 0, 0.0715, 0.0085, 0.001], dtype=torch.float32).view(1, -1).to(device)
    
    # Load the DDPM model
    ddpm = load_ddpm_model(model_path, device, n_classes=n_classes)
    
    # Generate and save images
    sample_images(ddpm, n_samples, image_size, c_i, guide_weight, save_dir, save_seq)