import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.ngpu = config["number_gpus"]
        nz = config["size_of_z_latent"] # 100
        ngf = config["number_of_generator_feature"] # 64
        nc = config["number_channels"] # 3
        # pylint: disable=bad-continuation
        # 反卷积 output = (input-1)stride+outputpadding -2padding+kernelsize, input 1, 4, 8,
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), # 4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), # 64
            nn.Tanh()
        )
        # pylint: enable=bad-continuation

    def forward(self, input):
        return self.main(input)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out #, attention


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DisEncoder(nn.Module):
    def __init__(self, config, isize, nz):
        super(DisEncoder, self).__init__()
        self.ngpu = config["number_gpus"]
        ndf = config["number_of_discriminator_feature"]
        nc = config["number_channels"]
        csize, cndf = isize / 2, ndf

        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf), nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf), nn.LeakyReLU(0.2, inplace=True))
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat), nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat), nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat), nn.LeakyReLU(0.2, inplace=True))
            # main.add_module('pyramid-{0}-selayer'.format(out_feat), SELayer(out_feat))
            cndf = cndf * 2

            csize = csize / 2
        main.add_module('final-{0}-{1}-conv'.format(cndf, 1), nn.Conv2d(cndf, nz, 4, 1, 0, bias=False)) # nz x 1 x 1
        # main.add_module('pyramid-{0}-selayer'.format(nz), SELayer(nz))
        self.main = main

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, config, isize, nz):
        super(Encoder, self).__init__()
        self.ngpu = config["number_gpus"]
        ndf = config["number_of_discriminator_feature"]
        nc = config["number_channels"]
        csize, cndf = isize / 2, ndf

        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf), nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf), nn.LeakyReLU(0.2, inplace=True))
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat), nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat), nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat), nn.ReLU(inplace=True))
            # main.add_module('pyramid-{0}-selayer'.format(out_feat), SELayer(out_feat))
            cndf = cndf * 2
            csize = csize / 2
        # main.add_module('pyramid-{0}-selfAtten'.format(cndf), Self_Attn(cndf,'relu'))
        main.add_module('final-{0}-{1}-conv'.format(cndf, 1), nn.Conv2d(cndf, nz, 4, 1, 0, bias=False)) # nz x 1 x 1
        # main.add_module('pyramid-{0}-selayer'.format(nz), SELayer(nz))
        self.main = main

    def forward(self, input):
        return self.main(input)

class Decoder(nn.Module):
    def __init__(self, config, isize):
        super(Decoder, self).__init__()
        self.ngpu = config["number_gpus"]
        nz = config["size_of_z_latent"] # 100
        ngf = config["number_of_generator_feature"] # 64
        nc = config["number_channels"] # 3

        csize = 4

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf), nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf), nn.ReLU(True))
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2), nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2
        main.add_module('final-{0}-{1}-convt'.format(cngf, nc), nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc), nn.Tanh())
        self.main = main

    def forward(self, input):
        return self.main(input)

class AAEGenerator(nn.Module):
    def __init__(self, config, isize):
        super(AAEGenerator, self).__init__()
        self.encoder = Encoder(config, isize, config["size_of_z_latent"])
        self.decoder = Decoder(config, isize)

    def forward(self, x):
        latent_i = self.encoder(x)
        gen_imag = self.decoder(latent_i)
        return gen_imag, latent_i

class AAEGenerator2(nn.Module):
    def __init__(self, config, isize):
        super(AAEGenerator2, self).__init__()
        self.encoder1 = Encoder(config, isize, config["size_of_z_latent"])
        self.decoder = Decoder(config, isize)
        self.encoder2 = Encoder(config, isize, config["size_of_z_latent"])

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder1(gen_imag)
        return gen_imag, latent_i, latent_o

class Discriminator(nn.Module):
    def __init__(self, config, isize):
        super(Discriminator, self).__init__()
        model = DisEncoder(config, isize, 1)
        layers = list(model.main.children())
        # self.features的内容为除了最后一层的前8层
        self.features = nn.Sequential(*layers[:-1])
        # wgan 最后一层
        self.last_layer = nn.Sequential(layers[-1])
        # 分类器
        # self.classifier = nn.Sequential(layers[-1])
        # self.classifier.add_module('Sigmoid', nn.Sigmoid())
        self.classifier = nn.Sigmoid()

    def forward(self, x):
        features = self.features(x)
        features = features
        last_layer = self.last_layer(features)
        classifier = self.classifier(last_layer)
        classifier = classifier.view(-1, 1).squeeze(1)
        last_layer1 = last_layer.mean(0)

        # classifier.type(torch.float32)

        return classifier, features , last_layer1.view(1), last_layer
    '''
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.ngpu = config["number_gpus"]
        ndf = config["number_of_discriminator_feature"]
        nc = config["number_channels"]
        # d = (d - kennel_size + 2 * padding) / stride + 1
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # 16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # 8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), # 4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    '''


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        # padding: 表示 四周 补0的个数, 卷积 权重 和 偏置 随机分配
        # 卷积核大小 （3，3）,  输入数据 四周 补 0 个数 为 1， 四周 补 一圈 0； 卷积之后， 原数据 长宽不变。
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 在通道上 归一化 ？ 理解不够深刻
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # BatchNorm2d 有学习参数 a,b
        # 两层 卷积层 都保持 输入大小不变
        self.stride = stride

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.relu(self.bn1(output))  # inplace 直接对传过来的值进行修改，不再经过中间变量； bn在激活函数之前

        output = self.conv2(output)
        output = self.bn2(output)

        output += residual  # 残差  # 图像大小 相同，才能相加
        return torch.relu(output)


class BaseRestBlock_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BaseRestBlock_Downsample, self).__init__()
        # 卷积， stride=2, 图像大小减半， 通道加倍
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 在通道上 归一化 ？ 理解不够深刻
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # BatchNorm2d 有学习参数 a,b
        # 两层 卷积层 都保持 输入大小不变
        self.stride = stride
        self.downsample = nn.Sequential(
            # 下采样， 不填充， 卷积核为1， 步长为2 -》 图像大小减半。  # 通过 卷积 来下采样， 图像减半 而不是 池化
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        residual = self.downsample(residual)  # 图像大小减半

        output = self.conv1(x)
        output = self.relu(self.bn1(output))

        output = self.conv2(output)
        output = self.bn1(output)

        output += residual
        return torch.relu(output)


class Resnet_18(nn.Module):
    def __init__(self, config, nc, nz):
        super(Resnet_18, self).__init__()
        # 卷积 (W-F+2p)/stride[取下] + 1
        nc = config["number_channels"]
        nz = config["size_of_z_latent"]
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3, bias=False)  # same 卷积  （stride=2）图像大小 减半
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 根据 池化核 补偿， 池化后 图像大小减半
        # 每层 两个 残差块
        self.layer1 = nn.Sequential(ResNetBasicBlock(nc, 64, 1),  # 残差块 图像大小 不变
                                    ResNetBasicBlock(64, 64, 1))  # (64, 64, (3, 3)) * 2

        self.layer2 = nn.Sequential(BaseRestBlock_Downsample(64, 128, [2, 1]),  # 图像大小 减倍
                                    ResNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(BaseRestBlock_Downsample(128, 256, [2, 1]),  # 通道加倍，图像大小减半
                                    ResNetBasicBlock(256, 256, 1))
        self.selfAttention1 = Self_Attn(256,'relu')
        self.layer4 = nn.Sequential(BaseRestBlock_Downsample(256, 512, [2, 1]),  # 通道加倍，图像大小减半
                                    ResNetBasicBlock(512, 512, 1))
        self.selfAttention2 = Self_Attn(512, 'relu')
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 平均池化 输出小 图像大小为 (1, 1)

        self.fc = nn.Linear(512, 1, bias=True)  # 平均池化（1，1）可以确定 输入个数
        self.last_sigmod = nn.Sigmoid()

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.selfAttention1(output)
        output = self.layer4(output)
        output = self.selfAttention2(output)
        output = self.avgpool(output)
        batch_size = output.shape[0]
        output = output.reshape(batch_size, -1)
        output = self.fc(output)
        last = self.last_sigmod(output)
        last = last.view(-1, 1).squeeze(1)
        return last, output.mean(0).view(1)
