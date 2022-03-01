import os
import time

import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.nn as nn

import model
import show
import record

class NNGraph(object):
    def __init__(self, dataloader, config, isize):
        super(NNGraph, self).__init__()
        self.config = config
        self.isize = isize
        self.train_model = self._get_train_model(config)
        record.record_dict(self.config, self.train_model["config"])
        self.config = self.train_model["config"]
        self.dataloader = dataloader


    def _get_train_model(self, config):
        train_model = model.init_train_model(config, self.isize)
        # train_model = self._load_train_model(train_model)
        return train_model

    def _save_train_model(self):
        model_dict = model.get_model_dict(self.train_model)
        file_full_path = record.get_check_point_file_full_path(self.config)
        torch.save(model_dict, file_full_path)

    def _load_train_model(self, train_model):
        '''
            path: save/"dataset_image_size"_"batch_size"_
            "number_of_generator_feature"_"number_of_discriminator_feature"_"size_of_z_latent"_"learn_rate"
            /checkpoint.tar
        '''
        file_full_path = record.get_check_point_file_full_path(self.config)
        if os.path.exists(file_full_path) and self.config["train_load_check_point_file"]:
            checkpoint = torch.load(file_full_path)
            train_model = model.load_model_dict(train_model, checkpoint)
        return train_model

    def _train_step(self, data, i):
        netG = self.train_model["netG"]
        optimizerG = self.train_model["optimizerG"]
        netD = self.train_model["netD"]
        optimizerD = self.train_model["optimizerD"]
        device = self.config["device"]

        # real_data = data[0].to(device)
        input = torch.empty(size=(self.config["batch_size"], 3, self.isize, self.isize), dtype=torch.float32,
                                 device=device)
        label = torch.empty(size=(self.config["batch_size"],), dtype=torch.float32, device=device)
        # gt = torch.empty(size=(self.config["batch_size"],), dtype=torch.long, device=device)
        real_label = torch.ones(size=(self.config["batch_size"],), dtype=torch.float32, device=device)
        fake_label = torch.zeros(size=(self.config["batch_size"],), dtype=torch.float32, device=device)


        with torch.no_grad():
            input.resize_(data[0].size()).copy_(data[0])
            # gt.resize_(data[1].size()).copy_(data[1])
            label.resize_(data[1].size())


        fake, latent_i = netG(input)
        _, latent_o= netG(fake)
        pred_real, feat_real, real_last = netD(input)
        pred_fake, feat_fake, fake_last = netD(fake.detach())

        errD = torch.tensor([0])

        if i % self.config["generator_learntimes"] == 0:
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
            for parm in netD.parameters():
                parm.data.clamp_(-self.config["clamp_num"], self.config["clamp_num"])
            errD = model.get_Discriminator_loss(netD, optimizerD, pred_real, pred_fake, real_label, fake_label, real_last, fake_last)
        errG = model.get_Generator_loss(netG, netD, optimizerG, input, fake, latent_i, latent_o, self.config)

        #if errD.item() < 1e-5:
            #self.train_model["netG"].apply(model._weights_init)

        return errD, errG

    '''
        noise = model.get_noise(real_data, self.config)
        fake_data = netG(noise)
        label = model.get_label(real_data, self.config)
        label = label.to(torch.float32)

        errD, D_x, D_G_z1 = model.get_Discriminator_loss(netD, optimizerD, real_data, fake_data.detach(), label,
                                                         criterion, self.config)
        errG, D_G_z2 = model.get_Generator_loss(netG, netD, optimizerG, fake_data, label, criterion, self.config)

        return errD, errG, D_x, D_G_z1, D_G_z2
    '''

    def _train_a_step(self, data, i, epoch):
        start = time.time()
        errD, errG = self._train_step(data, i)
        end = time.time()
        step_time = end - start

        self.train_model["take_time"] = self.train_model["take_time"] + step_time

        print_every = self.config["print_every"]
        if i % print_every == 0:
            record.print_status(step_time*print_every,
                                self.train_model["take_time"],
                                epoch,
                                i,
                                errD,
                                errG,
                                self.config,
                                self.dataloader)
        return errD, errG

    def _DCGAN_eval(self):
        # fixed_noise: 64, nz, 1, 1
        fixed_noise = self.train_model["fixed_noise"]
        with torch.no_grad():
            netG = self.train_model["netG"]
            fake = netG(fixed_noise).detach().cpu() # 64, nc, 64, 64
            return fake

    def _save_generator_images(self, iters, epoch, i):
        num_epochs = self.config["num_epochs"]
        save_every = self.config["save_every"]
        img_list = self.train_model["img_list"]

        if (iters % save_every == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):
            fake = self._DCGAN_eval() # 64, nc, 64, 64
            img_one = vutils.make_grid(fake, padding=2, normalize=True)
            img_list.append(img_one)
            show._show_one_img(img_one)
            self._save_train_model()

    def _train_iters(self):
        num_epochs = self.config["num_epochs"]
        G_losses = self.train_model["G_losses"]
        D_losses = self.train_model["D_losses"]
        iters = self.train_model["current_iters"]
        start_epoch = self.train_model["current_epoch"]

        # if self.config["add_gasuss"]:
            # for _, data in enumerate(self.dataloader['train'], 0):
                # data[0] = self.gasuss_noise(data[0], var=0.001)
        for epoch in range(start_epoch, num_epochs):
            self.train_model["current_epoch"] = epoch
            for i, data in enumerate(self.dataloader['train'], 0):
                errD, errG = self._train_a_step(data, i, epoch)
                G_losses[0].append(i + epoch * len(self.dataloader['train']))
                G_losses[1].append(errG.item())
                if errD.item() != 0:
                    D_losses[0].append(i + epoch * len(self.dataloader['train']))
                    D_losses[1].append(errD.item())
                iters += 1
                self.train_model["current_iters"] = iters

                # self._save_generator_images(iters, epoch, i)
            self.test()
        self._save_loss_images(G_losses, D_losses)

    def _save_loss_images(self, G_losses, D_losses):
        x1 = G_losses[0]
        x2 = D_losses[0]
        y1 = G_losses[1]
        y2 = D_losses[1]

        fig = plt.figure(figsize=(7, 5))  # figsize是图片的大小`
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig.add_subplot(2, 1, 1)  # ax1是子图的名字`
        plt.plot(x1, y1, 'g-', label=u'G_loss')
        plt.legend()  # 显示图例, 图例中内容由 label 定义
        plt.ylabel('loss')  # 横坐标轴的标题
        plt.xlabel('iters')  # 纵坐标轴的标题
        fig.add_subplot(2, 1, 2)  # ax1是子图的名字`
        plt.plot(x2, y2, 'r-', label=u'D_loss')
        plt.legend()  # 显示图例, 图例中内容由 label 定义
        plt.ylabel('loss')  # 横坐标轴的标题
        plt.xlabel('iters')  # 纵坐标轴的标题
        ticks = time.time()
        # plt.text(-1, -1, "generator_learntimes: %s\t " % self.config["generator_learntimes"])
        show._save_loss(G_losses, D_losses, ticks)
        plt.title('Loss图-mnist-1')  # 图形的标题

        # 显示图形
        plt.savefig("save/loss/loss_%s.png" % int(ticks))

        plt.show()

    def gasuss_noise(self, image, mean=0, var=0.001):
        '''
            添加高斯噪声
            image:原始图像
            mean : 均值
            var : 方差,越大，噪声越大
        '''
        # image = np.array(image / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
        noise = np.random.normal(mean, var ** 0.5, image.shape)  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
        out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
        # out = np.uint8(out * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复
        # cv.imshow("gasuss", out)
        # noise = noise*255
        return out

    def train(self):
        self._train_iters()

        # show.show_images(self.train_model, self.config, self.dataloader)


    def test(self):
        device = self.config["device"]

        with torch.no_grad():
            netG = self.train_model["netG"]

            an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                         device=device)
            gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,
                                         device=device)
            latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.config["size_of_z_latent"]), dtype=torch.float32,

                                        device=device)
            latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.config["size_of_z_latent"]), dtype=torch.float32,
                                        device=device)

            input = torch.empty(size=(self.config["batch_size"], 3, self.isize, self.isize), dtype=torch.float32,
                                device=device)
            label = torch.empty(size=(self.config["batch_size"],), dtype=torch.float32, device=device)
            gt = torch.empty(size=(self.config["batch_size"],), dtype=torch.long, device=device)
            for _, data in enumerate(self.dataloader['test'], 0):
                data[0] = self.gasuss_noise(data[0])
            time_i = time.time()
            if self.config["add_gasuss"]:
                for _, data in enumerate(self.dataloader['test'], 0):
                    data[0] = self.gasuss_noise(data[0], var=0.01)
            for i, data in enumerate(self.dataloader['test'], 0):
                with torch.no_grad():
                    input.resize_(data[0].size()).copy_(data[0])
                    gt.resize_(data[1].size()).copy_(data[1])
                    label.resize_(data[1].size())

                fake, latent_input = netG(input)
                _, latent_fake = netG(fake)

                # error = torch.zeros_like(fake)
                if self.config["score_method"] == "ganomaly":
                    error = torch.mean(torch.pow((latent_input - latent_fake), 2), dim=1)
                else:
                    err_d_con = torch.mean((fake - input), dim=1)
                    err_d_con = torch.mean(err_d_con, dim=[1, 2], keepdims=True)
                    err_d_enc = torch.mean(torch.pow((latent_input - latent_fake), 2), dim=1)
                    error = err_d_con * (1 - self.config["error_lamda"]) + \
                            err_d_enc * self.config["error_lamda"]

                an_scores[i * self.config["batch_size"]: i * self.config["batch_size"] + error.size(0)] = error.reshape(
                    error.size(0))
                gt_labels[i * self.config["batch_size"]: i * self.config["batch_size"] + error.size(0)] = gt.reshape(
                    error.size(0))
                latent_i[i * self.config["batch_size"]: i * self.config["batch_size"] + error.size(0),:] = latent_input.reshape(
                    error.size(0), self.config["size_of_z_latent"])
                latent_o[i * self.config["batch_size"]: i * self.config["batch_size"] + error.size(0),:] = latent_fake.reshape(
                    error.size(0), self.config["size_of_z_latent"])
                '''
                print_every = self.config["print_every"]
                
                if i % print_every == 0 or i == len(self.dataloader['test'].dataset) / self.config["batch_size"]:
                    img1 = vutils.make_grid(fake, padding=2, normalize=True)
                    img2 = vutils.make_grid(input, padding=2, normalize=True)
                    show._show_one_img(img1.cpu())
                    show._show_one_img(img2.cpu())
                '''
            an_scores = (an_scores - torch.min(an_scores)) / (torch.max(an_scores) - torch.min(an_scores))

            roc_auc = self.roc(gt_labels, an_scores, self.config)
            time_o = time.time()
            step_time = time_o - time_i
            record.print_scores(roc_auc, step_time, self.config)
            # record.save_status(self.config, print_str)
            # print(print_str)


    def roc(self, labels, scores, config, saveto = None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        labels = labels.cpu()
        scores = scores.cpu()

        # True/False Positive Rates.
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Equal Error Rate
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)




        if saveto:
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
            plt.plot([eer], [1 - eer], marker='o', markersize=5, color="navy")
            plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            # saveto = saveto + "_%s_%s_" % (config["mnist_abnormal_class"], time.strftime("%Y_%m_%d_%Hh_%Mm_%Ss", time.localtime()))
            # plt.savefig(os.path.join(saveto, 'ROC.pdf'))
            # plt.show()
            plt.close()


        return roc_auc

def test():
    ticks = time.time()
    ptr = "loss_%s" % int(ticks)
    print(ptr)

# test()