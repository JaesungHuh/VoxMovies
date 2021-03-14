#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tune_threshold import tuneThresholdfromScore
from datasetloader import test_dataset_loader


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, trainfunc, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module(
            'models.' + model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module(
            'loss.' + trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        return outp


class ModelTrainer(object):
    def __init__(self, speaker_model, **kwargs):

        self.__model__ = speaker_model

        self.gpu = 0

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self,
                         test_list,
                         test_path,
                         n_thread,
                         print_interval=100,
                         num_eval=10,
                         **kwargs):

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split()[-2:] for x in lines], [])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles,
                                           test_path,
                                           num_eval=num_eval,
                                           **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=n_thread,
            drop_last=False,
        )

        ## Extract features for every wavfiles
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inp1 = data[0][0].cuda()
                ref_feat = self.__model__(inp1).detach().cpu()
                feats[data[1][0]] = ref_feat
                telapsed = time.time() - tstart

                if idx % print_interval == 0:
                    sys.stdout.write(
                        "\rReading %d of %d: %.2f Hz, embedding size %d" %
                        (idx, len(setfiles), idx / telapsed,
                         ref_feat.size()[1]))

        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0, 1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__model__.module.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1),
                                       com_feat.unsqueeze(-1).transpose(
                                           0, 2)).detach().cpu().numpy()

            score = -1 * np.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz" %
                                 (idx, len(lines), idx / telapsed))
                sys.stdout.flush()

        print('')

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" %
                      (origname, self_state[name].size(),
                       loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
