# evaluate a smoothed classifier on a dataset
import argparse
import os
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core_gm import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
import numpy as np

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--K", type=int, default=10, help="number of normal dist in mixed gaussian")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier, map_location=torch.device('cpu'))
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    #loading dataset early to acquire data shape from x0
    dataset = get_dataset(args.dataset, args.split)
    (x0, label0) = dataset[0]

    #collecting random K means
    mu_batch = x0.repeat((args.K,1,1,1))
    print(mu_batch[0].size())
    mu_set = torch.randn_like(mu_batch, device='cpu') 
    sig_set = np.random.uniform(low = 0, high = args.sigma, size=args.K)
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, mu_set, sig_set)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    # DEFINE MEANS AND VARS HERE, FEED TO smoothed_classifier
    # MODIFY Smooth IN core.py
    # iterate through the dataset
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cpu() #example
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
