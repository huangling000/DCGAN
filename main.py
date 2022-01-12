import dataset
from etc import config
from graph import NNGraph

def run():
    isize = config["mnist_image_size"]
    dataloader = dataset.get_dataloader(config, 'mnist')
    g = NNGraph(dataloader, config, isize)
    g.train()

run()