import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import os 
import timm
    

def save_unnormalized_img(img, filename, data_config):
    inverse_transform = transforms.Normalize(mean=[-m/s for m, s in zip(data_config['mean'], data_config['std'])], std=[1/s for s in data_config['std']])
    inversed_image = inverse_transform(img)
    save_image(inversed_image, filename)


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


class DiffusionRobustModel(nn.Module):
    def __init__(self, ptfile, sample_output_imgs_folder):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("imageNet/256x256_diffusion_uncond.pt")
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 
        local_model_exists = os.path.isfile(ptfile)
        classifier = None
        if local_model_exists:
            print("Using local model", flush=True)
            classifier = torch.load(ptfile)
        else:
            print("Fetching remote model", flush=True)
            classifier = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=True)
            torch.save(model, ptfile)
        classifier.eval().cuda()

        self.classifier = classifier

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()

        self.data_config = None
        self.image_num = 0
        self.sample_output_imgs_folder = sample_output_imgs_folder

    def forward(self, x, t):
        if self.image_num < 15:
            for idx, img in enumerate(x):
                filename = self.sample_output_imgs_folder + f"/original_image_{self.image_num}.png"
                save_unnormalized_img(img, filename, self.data_config)
                break

        imgs = self.denoise(x, t)

        if self.image_num < 15:
            for idx, img in enumerate(imgs):
                filename = self.sample_output_imgs_folder + f"/denoised_image_{self.image_num}.png"
                save_unnormalized_img(img, filename, self.data_config)
                break

            self.image_num += 1

        imgs = torch.nn.functional.interpolate(imgs, (384, 384), mode='bilinear', antialias=True)
        imgs = imgs.cuda()
        with torch.no_grad():
            out = self.classifier(imgs)

        return out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
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