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
    def __init__(self, block, in_channels, out_channels, num_block, down_sample=True):
        super(encoder, self).__init__()

        self.encoder = nn.Sequential(
            *[block(in_channels, in_channels) for _ in range(num_block)]
        )

        if down_sample:
            self.encoder.append(down_sample_transition_block(in_channels, out_channels))
        
    def forward(self, input):
        return self.encoder(input)
    
class decoder(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_block, up_sample=True):
        super(decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            *[block(in_channels, in_channels) for _ in range(num_block)]
        )

        if up_sample:
            self.decoder.append(up_sample_transition_block(in_channels, out_channels))
        
    def forward(self, input):
        return self.decoder(input)
    
class DenseUNet(nn.Module):
    def __init__(self, block = conv_block, in_channels=3, out_channels=1, num_blocks=[2,2,2,2,2,2,2], mode = 'default'):
        super(DenseUNet, self).__init__()
        
        self.mode = mode
        n1 = 64
        filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32, n1 * 64]

        # self.encoder1 = encoder(block, in_channels, filter[0], num_blocks[0], down_sample=False)
        self.encoder1 = block(in_channels, filter[0])
        self.encoder2 = encoder(block, filter[0], filter[1], num_blocks[1])
        self.encoder3 = encoder(block, filter[1], filter[2], num_blocks[2])
        self.encoder4 = encoder(block, filter[2], filter[3], num_blocks[3])
        self.encoder5 = encoder(block, filter[3], filter[4], num_blocks[4])
        self.encoder6 = encoder(block, filter[4], filter[5], num_blocks[5])
        self.encoder7 = encoder(block, filter[5], filter[6], num_blocks[6])

        self.decoder1 = decoder(block, filter[6], filter[5], num_blocks[5])
        self.decoder2 = decoder(block, filter[5], filter[4], num_blocks[4])
        self.decoder3 = decoder(block, filter[4], filter[3], num_blocks[3])
        self.decoder4 = decoder(block, filter[3], filter[2], num_blocks[2])
        self.decoder5 = decoder(block, filter[2], filter[1], num_blocks[1])
        self.decoder6 = decoder(block, filter[1], filter[0], num_blocks[0])
        # self.decoder7 = decoder(block, filter[0], out_channels, num_blocks[0], up_sample=False)
        self.decoder7 = block(filter[0], out_channels)

        self.down_sample1 = down_sample_transition_block(filter[4], filter[5])
        self.down_sample2 = down_sample_transition_block(filter[3], filter[4])
        self.down_sample3 = down_sample_transition_block(filter[2], filter[3])
        self.down_sample4 = down_sample_transition_block(filter[1], filter[2])
        self.down_sample5 = down_sample_transition_block(filter[0], filter[1])

        self.up_sample1 = up_sample_transition_block(filter[5], filter[4])
        self.up_sample2 = up_sample_transition_block(filter[4], filter[3])
        self.up_sample3 = up_sample_transition_block(filter[3], filter[2])
        self.up_sample4 = up_sample_transition_block(filter[2], filter[1])
        self.up_sample5 = up_sample_transition_block(filter[1], filter[0])

        self.transform1 = nn.Conv2d(filter[6], filter[5], kernel_size=1, padding=0, bias=True)
        self.transform2 = nn.Conv2d(filter[5], filter[4], kernel_size=1, padding=0, bias=True)
        self.transform3 = nn.Conv2d(filter[4], filter[3], kernel_size=1, padding=0, bias=True)
        self.transform4 = nn.Conv2d(filter[3], filter[2], kernel_size=1, padding=0, bias=True)
        self.transform5 = nn.Conv2d(filter[2], filter[1], kernel_size=1, padding=0, bias=True)
        self.transform6 = nn.Conv2d(filter[1], filter[0], kernel_size=1, padding=0, bias=True)

    def forward(self, input):
        e1 = self.encoder1(input)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)

        if self.mode == 'default':
            d4 = self.decoder1(e5)
            d3 = self.decoder2(d4)
            d2 = self.decoder3(d3)
            d1 = self.decoder4(d2)
        elif self.mode == 'comcat':
            d4 = self.transform1(torch.cat((self.decoder1(e5),e4),dim=1))
            d3 = self.transform2(torch.cat((self.decoder2(d4),e3),dim=1))
            d2 = self.transform3(torch.cat((self.decoder3(d3),e2),dim=1))
            d1 = self.transform4(torch.cat((self.decoder4(d2),e1),dim=1))
        elif self.mode == 'idea':
            d6 = self.transform1(torch.cat((self.decoder1(e7),self.down_sample1(e5)-e6),dim=1))
            d5 = self.transform2(torch.cat((self.decoder2(d6),self.down_sample2(e4)-e5),dim=1))
            d4 = self.transform3(torch.cat((self.decoder3(d5),self.down_sample3(e3)-e4),dim=1))
            d3 = self.transform4(torch.cat((self.decoder4(d4),self.down_sample4(e2)-e3),dim=1))
            d2 = self.transform5(torch.cat((self.decoder5(d3),self.down_sample5(e1)-e2),dim=1))
            d1 = self.transform6(torch.cat((self.decoder6(d2),e1),dim=1))

        output = self.decoder7(d1)
        return output

# model = DenseUNet(mode='idea')
# input = torch.randn(1, 3, 320, 640)
# output = model(input)
# print(output.shape)