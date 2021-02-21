import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.conv_left1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            self.relu
        )
        
        self.conv_left2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            self.relu
        )

        self.conv_left3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            self.relu
        )

        self.conv_left4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            self.relu
        )

        self.conv_left5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            self.relu
        )

        self.conv_right1 = nn.Sequential(
            nn.Conv2d(128+256, 128, 3, padding=1),
            self.relu
        )

        self.conv_right2 = nn.Sequential(
            nn.Conv2d(64+128, 64, 3, padding=1),
            self.relu
        )

        self.conv_right3 = nn.Sequential(
            nn.Conv2d(32+64, 32, 3, padding=1),
            self.relu
        )

        self.conv_right4 = nn.Sequential(
            nn.Conv2d(16+32, 16, 3, padding=1),
            self.relu
        )

        self.conv_output = nn.Conv2d(16, 6, 1)


    def forward(self, x):
        """
        In:
            x: Tensor[batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output
        """
        # TODO
        x1 = self.conv_left1(x)
        x2 = self.max_pool_2x2(x1)
        x3 = self.conv_left2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.conv_left3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.conv_left4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.conv_left5(x8)

        x = F.interpolate(x9, scale_factor=2)
        x = torch.cat([x, x7], 1)
        x = self.conv_right1(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x5], 1)
        x = self.conv_right2(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x3], 1)
        x = self.conv_right3(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x1], 1)
        x = self.conv_right4(x)

        output = self.conv_output(x)
        
        return output

# Tried for extra credit
class UNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(UNet, self).__init__()
        # TODO
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.conv_left1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            self.relu,
            nn.Conv2d(16, 16, 3, padding=1),
            self.relu
        )
        
        self.conv_left2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            self.relu,
            nn.Conv2d(32, 32, 3, padding=1),
            self.relu
        )

        self.conv_left3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            self.relu,
            nn.Conv2d(64, 64, 3, padding=1),
            self.relu
        )

        self.conv_left4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            self.relu,
            nn.Conv2d(128, 128, 3, padding=1),
            self.relu
        )

        self.conv_left5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            self.relu,
            nn.Conv2d(256, 256, 3, padding=1),
            self.relu
        )

        self.conv_right1 = nn.Sequential(
            nn.Conv2d(128+256, 128, 3, padding=1),
            self.relu,
            nn.Conv2d(128, 128, 3, padding=1),
            self.relu
        )

        self.conv_right2 = nn.Sequential(
            nn.Conv2d(64+128, 64, 3, padding=1),
            self.relu,
            nn.Conv2d(64, 64, 3, padding=1),
            self.relu
        )

        self.conv_right3 = nn.Sequential(
            nn.Conv2d(32+64, 32, 3, padding=1),
            self.relu,
            nn.Conv2d(32, 32, 3, padding=1),
            self.relu
        )

        self.conv_right4 = nn.Sequential(
            nn.Conv2d(16+32, 16, 3, padding=1),
            self.relu,
            nn.Conv2d(16, 16, 3, padding=1),
            self.relu
        )

        self.conv_output = nn.Conv2d(16, 6, 1)


    def forward(self, x):
        """
        In:
            x: Tensor[batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output
        """
        # TODO
        x1 = self.conv_left1(x)
        x2 = self.max_pool_2x2(x1)
        x3 = self.conv_left2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.conv_left3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.conv_left4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.conv_left5(x8)

        x = F.interpolate(x9, scale_factor=2)
        x = torch.cat([x, x7], 1)
        x = self.conv_right1(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x5], 1)
        x = self.conv_right2(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x3], 1)
        x = self.conv_right3(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x1], 1)
        x = self.conv_right4(x)

        output = self.conv_output(x)
        
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
