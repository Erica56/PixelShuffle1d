import torch
from torch import nn

class PixelShuffle1d(nn.Module):
    """
    输入batch * channel * length
    输出batch * channel // upscaler * length X upscaler
    """
    def __init__(self, upscaler):
        super().__init__()
        self.upscaler = upscaler

    def forward(self, x):
        batch_size, channels, length = x.shape
        results = torch.Tensor([])  # 存放最终结果
        for one_data in x:
            one_data = one_data.transpose(-1, -2)  # 变为 length * channels
            r_one_data = torch.Tensor([])  # 存放每条数据的结果
            for one_channel in one_data:
                one_channel = one_channel.view(-1, self.upscaler)
                # 变为了out_channels * upscaler 可以直接在输出通道维度上拼接
                r_one_data = torch.cat([r_one_data, one_channel], dim=1)
            r_one_data = r_one_data.unsqueeze(0)
            results = torch.cat([results, r_one_data], dim=0)
        return results


x = torch.rand(size=(1, 4, 6))

pixel = PixelShuffle1d(2)
y = pixel(x)
print(x.shape)
print(y.shape)
print(x)
print(y)
