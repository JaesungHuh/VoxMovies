#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import argparse
import torch
from tune_threshold import *
from speakernet import *

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description='SpeakerNet')

## Data loader
parser.add_argument('--max_frames',
                    type=int,
                    default=200,
                    help='Input length to the network for training')
parser.add_argument(
    '--eval_frames',
    type=int,
    default=300,
    help='Input length to the network for testing; 0 uses the whole files')
parser.add_argument('--batch_size',
                    type=int,
                    default=200,
                    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk',
                    type=int,
                    default=500,
                    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--n_thread',
                    type=int,
                    default=4,
                    help='Number of loader threads')

## Loss functions
parser.add_argument(
    '--nClasses',
    type=int,
    default=5994,
    help=
    'Number of speakers in the softmax layer, only for softmax-based losses')
## Load and save
parser.add_argument('--initial_model',
                    type=str,
                    default='',
                    help='Initial model weights')
## Training and test data
parser.add_argument('--test_list',
                    type=str,
                    default='data/test_list.txt',
                    help='Evaluation list')
parser.add_argument('--test_path',
                    type=str,
                    default='data/voxceleb1',
                    help='Absolute path to the test set')
parser.add_argument('--trainfunc',
                    type=str,
                    default='softmaxproto',
                    help='train function')
## Model definition
parser.add_argument('--n_mels',
                    type=int,
                    default=40,
                    help='Number of mel filterbanks')
parser.add_argument('--log_input',
                    type=bool,
                    default=False,
                    help='Log input features')
parser.add_argument('--model',
                    type=str,
                    default='',
                    help='Name of model definition')
parser.add_argument('--encoder_type',
                    type=str,
                    default='ASP',
                    help='Type of encoder')
parser.add_argument('--nOut',
                    type=int,
                    default=512,
                    help='Embedding size in the last FC layer')

args = parser.parse_args()

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Trainer script
# ## ===== ===== ===== ===== ===== ===== ===== =====


def evaluate_model(args):
    ## Load models
    s = SpeakerNet(**vars(args))
    s = WrappedModel(s).cuda(0)

    it = 1

    ## Initialise trainer and data loader
    trainer = ModelTrainer(s, **vars(args))

    if (args.initial_model != ''):
        trainer.loadParameters(args.initial_model)
        print('Model %s loaded!' % args.initial_model)
    else:
        print('Please specify model_path for evaluation')
        sys.exit()

    ## Evaluation code - must run on single GPU
    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

    print('Total parameters: ', pytorch_total_params)
    print('Test list', args.test_list)

    sc, lab, _ = trainer.evaluateFromList(**vars(args))
    result = tuneThresholdfromScore(sc, lab, [1, 0.1])

    p_target = 0.05
    c_miss = 1
    c_fa = 1

    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss,
                                      c_fa)

    print('EER %2.4f MinDCF %.5f' % (result[1], mindcf))


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())

    evaluate_model(args)


if __name__ == '__main__':
    main()
