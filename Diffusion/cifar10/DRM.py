import torch
import torch.nn as nn 
from models.resnet50 import ResNet50
from torchvision.utils import save_image
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from transformers import AutoModelForImageClassification


class Args:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=4000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


class DiffusionRobustModel(nn.Module):
    def __init__(self, filename, sample_output_imgs_folder):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("cifar10/cifar10_uncond_50M_500K.pt")
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 

        classifier = torch.load(filename)
        classifier.eval().cuda()
        self.image_num = 0
        self.classifier = classifier
        self.sample_output_imgs_folder = sample_output_imgs_folder

    def forward(self, x, t, y):
        d_imgs = self.denoise(x, t)

        d_imgs = torch.nn.functional.interpolate(d_imgs, (224, 224), mode='bilinear', antialias=True)
        d_imgs = d_imgs.cuda()

        p_imgs = torch.nn.functional.interpolate(x, (224, 224), mode='bilinear', antialias=True)
        with torch.no_grad():
            d_out = self.classifier(d_imgs)
            p_out = self.classifier(p_imgs)
            
            # Save denoised image and prediction:
            _, d_classes = torch.max(d_out[1].data, 1)
            d_class = d_classes[0]

            if self.image_num < 15:
                for idx, img in enumerate(d_imgs):
                    filename = self.sample_output_imgs_folder + f"/denoised_image_{self.image_num}_pred_{d_class}_true_{y[idx]}.png"
                    break

            # Save pgd image and prediction:
            _, p_classes = torch.max(p_out[1].data, 1)
            p_class = p_classes[0]

            if self.image_num < 15:
                for idx, img in enumerate(p_imgs):
                    filename = self.sample_output_imgs_folder + f"/pgd_image_{self.image_num}_pred_{p_class}_true_{y[idx]}.png"
                    save_image(img, filename)
                    break

                self.image_num += 1
        
        return d_out, p_out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out