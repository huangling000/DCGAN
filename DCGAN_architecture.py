import torch
import torch.nn as nn

from SAN import Subtraction, Aggregation, Subtraction2


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    print(loc_h)
    print(loc_w)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    print(loc)
    return loc

class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0: # pairwise
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else: # patchwise
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x

class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out

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
            # main.add_module('pyramid-{0}-batchnorm'.format(out_feat), nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat), nn.LeakyReLU(0.2, inplace=True))
            # main.add_module('pyramid-{0}-selayer'.format(out_feat), SELayer(out_feat))
            # main.add_module('pyramid-{0}-SelfAtten'.format(out_feat), Self_Attn(out_feat, 'relu'))
            main.add_module('pyramid-{0}-SAN'.format(out_feat), Bottleneck(1, out_feat, out_feat // 16, out_feat // 4, out_feat, 8, 3))
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
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        last_layer = self.last_layer(features)
        last_layer = last_layer.mean(0)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        # classifier.type(torch.float32)

        return classifier, features, last_layer.view(1)
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