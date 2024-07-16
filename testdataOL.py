

from libs.utils.config import Config
import torch.utils.data as data
from libs.dataset.openlane.perprocess import Preprocessing


def loadDataOL():
    opt = Config.fromfile('./options4OL.py')
    from libs.dataset.openlane.datasetOL import Dataset_TrainV1, multibatch_collate_fn
    dataset = Dataset_TrainV1(cfg=opt.dscfg, mode='training')
    testloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
                            collate_fn=multibatch_collate_fn)
    for batch in testloader:
        print(type(batch))

def loadDataOLV2():
    opt = Config.fromfile('./options4OLV2.py')
    from libs.dataset.openlane.datasetOLV2 import Dataset_TrainV2, multibatch_collate_fn
    dataset = Dataset_TrainV2(cfg=opt.dscfg, mode='training')
    # testloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
    #                         collate_fn=multibatch_collate_fn)
    # for batch in testloader:
    #     print(type(batch))
    for input in dataset:
        print(type(input))

if __name__ == '__main__':
    # loadDataOL()
    # loadDataOLV2()

    opt = Config.fromfile('./options4OLV2.py')
    dataprocessing = Preprocessing(cfg=opt.dscfg)
    dataprocessing.run()
    