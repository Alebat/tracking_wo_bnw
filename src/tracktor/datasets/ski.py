import configparser
import csv
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2
import pandas as pd

from ..config import cfg
from torchvision.transforms import ToTensor


def read_images(vc, rotate90=False):
    yes = True
    f = 0
    while yes:
        yes, img = vc.read()
        if yes:
            if rotate90:
                yield f, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                yield f, img
            f += 1


class SkiSequence(Dataset):
    def __init__(self, seq_name):
        """
        Args:
            seq_name (string): Sequence to take
        """
        self._seq_name = seq_name
        self.no_gt = True

        self._mot_dir = osp.join(cfg.DATA_DIR, 'Ski')

        self._folders = os.listdir(self._mot_dir)

        self.transforms = ToTensor()

        assert seq_name in self._folders, \
            'Image set does not exist: {}'.format(seq_name)

        self._data, self._det, self._len = self._sequence()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        f, img = next(self._data)
        img = self.transforms(img)
        print(f)
        if f in self._det.index:
            dets = self._det.loc[[f]]
            conv = np.array([det.astype(np.float32) for det in dets.values[:, 1:5]])
        else:
            conv = np.ndarray((0,))

        sample = {
            'img': img,
            'dets': conv,
        }

        return sample

    def _sequence(self):
        seq_name = self._seq_name
        seq_path = osp.join(self._mot_dir, seq_name)

        vc = cv2.VideoCapture(seq_path)
        length = int(vc.get(7))

        images = read_images(vc)

        det = pd.read_csv(f"{seq_path}.txt",
                          sep=',',
                          header=None,
                          index_col=0,
                          names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])

        return images, det, length

    def __str__(self):
        return f"Ski-{self._seq_name}"

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, self._seq_name + '.tracks.txt')

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])


class SkiWrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split='all'):
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        """

        self._mot_dir = osp.join(cfg.DATA_DIR, 'Ski')
        self._folders = os.listdir(os.path.join(self._mot_dir))

        sequences = self._folders

        if not 'all' == split:
            if split in sequences:
                sequences = [split]
            else:
                raise NotImplementedError("Ski split not available " + split)

        self._data = []
        for s in sequences:
            self._data.append(SkiSequence(seq_name=s))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
