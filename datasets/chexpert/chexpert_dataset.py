import numpy as np
import torch
import os

from datasets.deprecated.chexpert.data.single_disease import SingleDiseaseDataset
# from datasets.chexpert.chexpert_labels import race_dict_inv


class ChexpertSensitive(SingleDiseaseDataset):
    def __init__(self, csv, config_path, mode):
        # super(ChexpertSensitive, self).__init__(in_csv_path=csv, cfg=config_path)
        in_csv_path=csv
        cfg=config_path
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self.race = []
        self._mode = mode
        # self.dict = {'1.0': 1, '': 0, '-1.0': 2, '0.0': 3}
        # self.dict = cfg.class_mapping
        self.dict = [{'1.0': 1, '': 0, '0.0': 0, '-1.0': 0},
                     {'1.0': 1, '': 0, '0.0': 0, '-1.0': 1}, ]
        
        race_dict = {
            0 : 'White',
            #1 : 'Black',
            1 : 'Asian',
        }
        # and the inverse key-value mapping
        race_dict_inv = dict((v,k) for k,v in race_dict.items())

        with open(cfg.dataset_path + in_csv_path) as f:
            header = f.readline().strip('\n').split(',')
            index = None
            for nr, disease in enumerate(header):
                if disease == cfg.disease:
                    index = nr
                    break
            if index is None:
                raise Exception(
                    f'Disease {cfg.disease} was not found in the dataset.')
            else:
                print('index: ', index)
            assert header[index] == cfg.disease
            self._label_header = [header[index]]
            cfg._label_header = self._label_header
            for line in f:
                labels = []
                races = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                flg_enhance = False
                value = fields[index]
                race = fields[-1]
                if index == 10 or index == 13:
                    labels.append(self.dict[1].get(value))
                    if self.dict[1].get(
                            value) == '1' and \
                            self.cfg.enhance_index.count(index) > 0:
                        flg_enhance = True
                elif index == 7 or index == 11 or index == 15:
                    labels.append(self.dict[0].get(value))
                    if self.dict[0].get(
                            value) == '1' and \
                            self.cfg.enhance_index.count(index) > 0:
                        flg_enhance = True

                # get full path to the image
                data_path_split = cfg.dataset_path.split('/')
                image_path_split = image_path.split('/')
                data_path_split = [x for x in data_path_split if x != '']
                image_path_split = [x for x in image_path_split if x != '']
                if data_path_split[-1] == image_path_split[0]:
                    full_image_path = data_path_split[:-1] + image_path_split
                    full_image_path = "/" + "/".join(full_image_path)
                else:
                    full_image_path = cfg.dataset_path + image_path

                assert os.path.exists(full_image_path), full_image_path

                self._image_paths.append(full_image_path)
                self._labels.append(labels)
                races.append(race_dict_inv[race])
                self.race.append(races)

                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(full_image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)
    
    def __getitem__(self, index):
        X, target = super(ChexpertSensitive, self).__getitem__(index)
        # data, target, sensitive attribute
        return X, target, np.array(self.race[index])