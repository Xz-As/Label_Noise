import torch
from torch import nn
from torchvision.models import resnet18

class CNN_(nn.Module):
    def __init__(self, num_classes = 10, n_blocks = 3, in_channels = 3, channels = (32, 64, 128), kernel_size = (4, 3, 4), strides = (3, 2, 2), padding = 3):
        super(CNN_, self).__init__()
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = channels[0],
                    kernel_size = kernel_size[0],
                    stride = strides[0],
                    padding = padding
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels = channels[0],
                    out_channels = channels[1],
                    kernel_size = kernel_size[1],
                    stride = strides[1],
                    padding = padding
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
                nn.Conv2d(
                    in_channels = channels[1],
                    out_channels = channels[2],
                    kernel_size = kernel_size[2],
                    stride = strides[2],
                    padding = padding
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )


        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features = channels[-1], out_features = num_classes)
        )

    def forward(self, x):
        feature = x
        feature = self.conv1(feature)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        output = self.fc(feature.view(feature.size(0), -1))

        return output


class ResidualBlock(nn.Module):
    """
    This is an object of residual block for resnet20.
    Args:
        in_channel (int): Input channels of the residual block.
        out_channel (int): Output channels of the residual block.
        stride (int): Stride of the residual block.
    """
    def __init__(self, in_channel, out_channel, stride = 1):
        super(ResidualBlock, self).__init__()
        # Main path
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel)
        )
        
        # Short cut
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()
        
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        feature = self.left(x)
        feature += self.shortcut(x)
        out = self.relu(feature)
        return out

class ResNet20(nn.Module):
    """
    This is an object of resnet20.
    Args:
        ResidualBlock (object): The basic residual block object.
        num_classes (int): Number of classes.
        n_blocks (int): Number of blocks.
        in_channels (int): Original input channels.
        channels (tuple): Output channels of each residual block.
        strides (tuple): Strides of each residual block.
    """
    def __init__(self, ResidualBlock = ResidualBlock, device = torch.device('cpu'), num_classes = 10, n_blocks = 3, in_channels = 3, channels = (16, 32, 64), strides = (1, 2, 2)):
        super(ResNet20, self).__init__()
        self.channels = channels
        self.strides = strides
        self.n_blocks = n_blocks
        self.device = device
        # Input layse
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(),
        )
        self.in_channels = self.channels[0]

        # Res locks
        self.layers = []
        for i in range(n_blocks):
            self.layers.append(self._make_layer(ResidualBlock, i))
        # Output layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)


    def _make_layer(self, block, i):
        
        strides = [self.strides[i]] + [1] * (self.n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, self.channels[i], stride).to(self.device))
            self.in_channels = self.channels[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        feature = self.conv1(x)
        for layer in self.layers:
            feature = layer(feature)
        feature = self.avg_pool(feature)
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)
        return output


# Define a Multi layer percetron network.
class FCN_(nn.Module):
    def __init__(self, in_channels = 28*28, hid_dim = 100, num_classes = 3):
        super(FCN_, self).__init__()
        self.input_layer = nn.Linear(in_channels, hid_dim)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hid_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        out = self.relu(self.input_layer(x.reshape(x.size(0), -1)))
        out = self.relu(self.hidden_layer(out))
        out = self.output_layer(out)
        
        return out
        
    

def FCN(**hyperparams):
    """ This function returns a FCN object.
    Args:
        in_dim(int): The input dimension
        hid_dim(int): The hidden dimension
        out_dim(int): The output dimension
    Returns:
        model(torch.nn.module): FCN object
    """
    
    model = FCN_(num_classes = hyperparams['num_classes'], in_channels = hyperparams['channels'] * hyperparams['img_size'])
    
    return model


def CNN(**hyperparams):
    """
    This function returns a CNN object
    Args:
        num_classes (int): The number of classes.
        in_channels (int): The original channels of data.
    Returns:
        model (torch.nn.module): CNN object
    """
    return CNN_(num_classes = hyperparams['num_classes'], in_channels = hyperparams['channels'])


def Resnet20(**hyperparams):
    """
    This function returns a resnet20 object for CIFAR dataset
    Args:
        num_classes (int): The number of classes.
        in_channels (int): The original channels of data.
    Returns:
        model (torch.nn.module): Resnet20 object
    """
    return ResNet20(ResidualBlock, num_classes = hyperparams['num_classes'], in_channels = hyperparams['channels'], device = hyperparams['device'])


def Resnet18(**hyperparams):
    """
    This function returns a resnet18
    Args:
        num_classes (int): The number of classes.
        in_channels (int): The original channels of data.
    Returns:
        model (torch.nn.module): Resnet18 object
    """
    # Predefined Resnet18 model without pretrained
    model = resnet18(pretrained = False, num_classes = hyperparams['num_classes'])
    model.conv1= nn.Conv2d(hyperparams['channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


# Demo
if __name__ == '__main__':
    DEVICE_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    img = torch.rand(200, 1, 28, 28).to(torch.device('cuda:0'))
    img_gt = torch.randn(200).to(torch.device('cuda:0'))
    hyp = {'channels': img.size(1), 'num_classes': 10, 'device': torch.device('cuda'), 'img_size': 28 * 28}
    
    net = FCN(**hyp)
    net = net.cuda()
    net.eval()
    for i in range(int((len(img) - 1) / batch_size) + 1):
        if (i + 1) * batch_size >= len(img):
            xs = img[i * batch_size :]
            ys = img_gt[i * batch_size :]
        else:
            xs = img[i * batch_size : (i + 1) * batch_size]
            ys = img_gt[i * batch_size : (i + 1) * batch_size]
        out = net(xs)
        print(out.size())