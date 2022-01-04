import torch

config = {}

config["dataset"] = "celebA"
config["abnormal_class"] = "1"
config["batch_size"] = 128
config["image_size"] = 64
config["mnist_image_size"] = 64
config["num_epochs"] = 5
config["data_path"] = "data/%s" % config["dataset"]
config["workers"] = 0
config["print_every"] = 200
config["save_every"] = 500
config["manual_seed"] = 999
config["train_load_check_point_file"] = False
#config["manual_seed"] = random.randint(1, 10000) # use if you want new results

config["number_channels"] = 3
config["size_of_z_latent"] = 100
config["number_gpus"] = 1
config["number_of_generator_feature"] = 64
config["number_of_discriminator_feature"] = 64
config["learn_rate"] = 0.0002
config["beta1"] =0.5
config["real_label"] = 1
config["fake_label"] = 0
config["device"] = torch.device("cuda:0" if (torch.cuda.is_available() and config["number_gpus"] > 0) else "cpu")