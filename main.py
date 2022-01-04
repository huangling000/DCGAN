import dataset
from etc import config
from graph import NNGraph

def run():
    dataloader = dataset.get_dataloader(config)
    g = NNGraph(dataloader, config)
    g.train()

run()