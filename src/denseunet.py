import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, input):
        return self.block(input)

class down_sample_transition_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_sample_transition_block, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input):
        return self.block(input)

class up_sample_transition_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_sample_transition_block, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, input):
        return self.block(input)

class encoder(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_block):
        super(encoder, self).__init__()

        self.encoder = nn.Sequential(
            *[block(in_channels, in_channels) for _ in range(num_block)],
            down_sample_transition_block(in_channels, out_channels)
        )
        
    def forward(self, input):
        return self.encoder(input)
    
class decoder(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_block):
        super(decoder, self).__init__()

        self.decoder = nn.Sequential(
            *[block(in_channels, in_channels) for _ in range(num_block)],
            up_sample_transition_block(in_channels, out_channels)
        )
        
    def forward(self, input):
        return self.decoder(input)
    
class DenseUNet(nn.Module):
    def __init__(self, block = conv_block, in_channels=3, out_channels=2, num_blocks=[2,2,2,2,2], idea_mode = False):
        super(DenseUNet, self).__init__()
        
        self.idea_mode = idea_mode
        n1 = 64
        filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.encoder1 = encoder(block, in_channels, filter[0], num_blocks[0])
        self.encoder2 = encoder(block, filter[0], filter[1], num_blocks[1])
        self.encoder3 = encoder(block, filter[1], filter[2], num_blocks[2])
        self.encoder4 = encoder(block, filter[2], filter[3], num_blocks[3])
        self.encoder5 = encoder(block, filter[3], filter[4], num_blocks[4])

        self.decoder1 = decoder(block, filter[4], filter[3], num_blocks[4])
        self.decoder2 = decoder(block, filter[3], filter[2], num_blocks[3])
        self.decoder3 = decoder(block, filter[2], filter[1], num_blocks[2])
        self.decoder4 = decoder(block, filter[1], filter[0], num_blocks[1])
        self.decoder5 = decoder(block, filter[0], out_channels, num_blocks[0])

    def forward(self, input):
        e1 = self.encoder1(input)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        if not self.idea_mode:
            d4 = self.decoder1(e5)
            d3 = self.decoder2(d4)
            d2 = self.decoder3(d3)
            d1 = self.decoder4(d2)
        else:
            None
            # 这部分还在写

        output = self.decoder5(d1)
        return output

# model = DenseUNet()
# input = torch.randn(1, 3, 320, 640)
# output = model(input)
# print(output.shape)