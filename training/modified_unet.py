import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.utils import BaseOutput


# Definir manualmente `UNet2DConditionOutput`
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor  # La salida esperada por el scheduler


class ModifiedUNet(UNet2DConditionModel):
    def __init__(self, original_unet):
        super().__init__(**original_unet.config)
        self.original_unet = original_unet

        # Obtener el número de canales de salida esperado de la UNet original
        out_channels = original_unet.config.out_channels

        # Nueva capa convolucional con el mismo número de canales de entrada/salida
        self.new_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.xavier_uniform_(self.new_conv.weight)

        # Normalización para mejorar estabilidad
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, latent, timestep, encoder_hidden_states, **kwargs):
        # Pasamos el input original a la UNet de Stable Diffusion
        latent_output = self.original_unet(
            latent, timestep, encoder_hidden_states, **kwargs
        )

        # Si la salida es un `tuple`, extraemos el primer elemento
        if isinstance(latent_output, tuple):
            latent_output = latent_output[0]

        # Si sigue siendo un `UNet2DConditionOutput`, extraemos `.sample`
        if hasattr(latent_output, "sample"):
            latent_output = latent_output.sample

        # Aplicamos la convolución pero como una modificación residual
        modified_latent = latent_output + self.new_conv(latent_output) * 0.1

        return UNet2DConditionOutput(sample=modified_latent)
