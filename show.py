import time

import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import  rcParams

from matplotlib.animation import ImageMagickWriter

import record

rcParams["animation.embed_limit"] = 500

def show_some_batch(real_batch,device):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

def _plot_real_and_fake_images(real_batch, device, img_list, save_path):

    plt.figure(figsize=(30, 30))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    name = "real_and_fake.jpg"
    full_path_name = "%s/%s" % (save_path, name)
    plt.savefig(full_path_name)
    #plt.show()

def _save_loss(G_losses, D_losses, ticks):
    with open("save/loss/loss_%s.txt" % int(ticks), "w") as f:
        f.writelines(str(G_losses[0]))
        f.writelines('\n')
        f.writelines(str(G_losses[1]))
        f.writelines('\n')
        f.writelines(str(D_losses[0]))
        f.writelines('\n')
        f.writelines(str(D_losses[1]))

def _read_loss(ticks):
    with open("save/loss/loss_%s.txt" % int(ticks), "r") as f:
        file = f.readlines()
    string = file[0].rstrip('\n')
    G_Losses0 = string[1 : len(string) - 1].split(',')
    string = file[1].rstrip('\n')
    G_Losses1 = string[1 : len(string) - 1].split(',')
    string = file[2].rstrip('\n')
    D_Losses0 = string[1 : len(string) - 1].split(',')
    string = file[3].rstrip('\n')
    D_Losses1 = string[1 : len(string) - 1].split(',')
    print(list(map(float, G_Losses0))[0])
    print(list(map(float, G_Losses1)))
    print(list(map(float, D_Losses0)))
    print(list(map(float, D_Losses1)))

def _show_generator_images(G_losses, D_losses, save_path):
    plt.figure(figsize=(40, 20))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    name = "G_D_losses.jpg"
    full_path_name = "%s/%s" % (save_path, name)
    plt.savefig(full_path_name)
    plt.show()

def _show_img_list(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
    plt.show()

def _show_one_img(img):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [plt.imshow(np.transpose(img, (1, 2, 0)), animated=True)]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.show()

def _save_img_list(img_list, save_path, config):
    _show_img_list(img_list)
    metadata = dict(title='generator images', artist='Matplotlib', comment='Movie support!')
    writer = ImageMagickWriter(fps=1,metadata=metadata)
    ims = [np.transpose(i, (1, 2, 0)) for i in img_list]
    fig, ax = plt.subplots()
    with writer.saving(fig, "%s/img_list.gif" % save_path,500):
        for i in range(len(ims)):
            ax.imshow(ims[i])
            ax.set_title("step {}".format(i * config["save_every"]))
            writer.grab_frame()

def show_images(train_model, config, dataloader):
    G_losses = train_model["G_losses"]
    D_losses = train_model["D_losses"]
    img_list = train_model["img_list"]
    save_path = record.get_check_point_path(config)

    _show_generator_images(G_losses, D_losses, save_path)
    _save_img_list(img_list,save_path,config)
    real_batch = next(iter(dataloader))
    _plot_real_and_fake_images(real_batch, config["device"], img_list, save_path)

