import torch

config = {}

config["dataset"] = "celebA"
config["mnist_abnormal_class"] = 1
config["cifar10_abnormal_class"] = 'car'
config["batch_size"] = 64
config["image_size"] = 64
config["mnist_image_size"] = 32
config["cifar10_image_size"] = 64
config["num_epochs"] = 1
config["generator_learntimes"] = 5
config["data_path"] = "data/%s" % config["dataset"]
config["workers"] = 0
config["print_every"] = 200
config["save_every"] = 500
config["manual_seed"] = 999
config["train_load_check_point_file"] = False
config["add_noise"] = True
config["clamp_num"] = 0.01
config["score_method"] = 'ganomaly' # 'ganomaly' or 'normal'
config["optimAdam"] = True
config["add_gasuss"] = True
config["Diters"] = 100
#config["manual_seed"] = random.randint(1, 10000) # use if you want new results

config["w_adv"] = 1
config["w_con"] = 50
config["w_enc"] = 1
config["error_lamda"] = 0.9
config["number_channels"] = 1
config["size_of_z_latent"] = 100
config["number_gpus"] = 1
config["number_of_generator_feature"] = 64
config["number_of_discriminator_feature"] = 64
config["generator_learn_rate"] = 0.0002
config["discriminator_learn_rate"] = 0.0002
config["beta1"] =0.5
config["real_label"] = 1
config["fake_label"] = 0
config["device"] = torch.device("cuda:0" if (torch.cuda.is_available() and config["number_gpus"] > 0) else "cpu")