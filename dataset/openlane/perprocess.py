import os
from libs.dataset.openlane.utils import load_pickle, save_pickle

class Preprocessing(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_video_datalist_for_processing(self):
        path = '/home/chengzy/MMA-TR-Net/dataset/OpenLane/OpenLane-V/list/datalist_video_training'
        datalist_video = load_pickle(path)

        datalist_out = dict()
        video_list = list(datalist_video) #622
        num = 0
        for i in range(len(video_list)): #循环video
            video_name = video_list[i]
            file_list = datalist_video[video_name]
            for j in range(len(file_list)): #video内遍历
                name = file_list[j] #segment-10017090168044687777_6380_000_6400_000_with_camera_labels/155008347084538900
                dirname = os.path.dirname(name) #segment-10017090168044687777_6380_000_6400_000_with_camera_labels
                filename = os.path.basename(name) #155008347084538900

                datalist_out[name] = list()
                datalist_out[name].append(name)
                for t in range(1, self.cfg.clip_length * 3): #难道这里*3的原因是为了做大限度让prev_filename in datalist
                    if j - t < 0:#如果没有past frame则跳出循环
                        break
                    prev_filename = file_list[j-t]
                    if len(datalist_out[name]) == self.cfg.clip_length + 1:
                        break
                    datalist_out[name].append(prev_filename)

                print(f'{num} ==> {filename} done')
                num += 1
                if len(datalist_out[name]) < self.cfg.clip_length + 1:
                    datalist_out.pop(name)
                    num -= 1

        print(f'The number of datalist_video: {len(datalist_out)}')
        save_pickle(f'/home/chengzy/MMA-TR-Net/dataset/OpenLane/OpenLane-V/list/datalist_training_{self.cfg.clip_length}', data=datalist_out)

    def run(self):
        print('start..........')
        self.get_video_datalist_for_processing()

