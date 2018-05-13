# Important: python 2.7 is required to run this script!

from __future__ import print_function

import sys
import re
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def main():
    log_files = process_arguments(sys.argv)

    train_iteration = []
    train_loss = []
    test_iteration = []
    test_loss = []
    test_accuracy = []

    acc_nocrf = []
    mean_recall_nocrf = []
    mean_jaccard_nocrf = []

    acc_crf = []
    mean_recall_crf = []
    mean_jaccard_crf = []

    base_test_iter = 0
    base_train_iter = 0

    for log_file in log_files:
        with open(log_file, 'rb') as f:
            if len(train_iteration) != 0:
                base_train_iter = train_iteration[-1]
                base_test_iter = test_iteration[-1]

            for line in f:
                # TRAIN NET
                if strstr(line, 'Iteration') and strstr(line, 'loss'):
                    matched = match_iteration(line)
                    train_iteration.append(int(matched.group(1)) + base_train_iter)
                    matched = match_loss(line)
                    train_loss.append(float(matched.group(1)))

                elif strstr(line, 'output #0: accuracy'):
                    matched = re.search(r'output #0: accuracy = (.*)', line)
                    acc_nocrf.append(float(matched.group(1)))

                elif strstr(line, 'output #1: accuracy'):
                    matched = re.search(r'output #1: accuracy = (.*)', line)
                    mean_recall_nocrf.append(float(matched.group(1)))

                elif strstr(line, 'output #2: accuracy'):
                    matched = re.search(r'output #2: accuracy = (.*)', line)
                    mean_jaccard_nocrf.append(float(matched.group(1)))

                elif strstr(line, 'output #3: accuracy'):
                    matched = re.search(r'output #3: accuracy_2 = (.*)', line)
                    acc_crf.append(float(matched.group(1)))

                elif strstr(line, 'output #4: accuracy'):
                    matched = re.search(r'output #4: accuracy_2 = (.*)', line)
                    mean_recall_crf.append(float(matched.group(1)))

                elif strstr(line, 'output #5: accuracy'):
                    matched = re.search(r'output #5: accuracy_2 = (.*)', line)
                    mean_jaccard_crf.append(float(matched.group(1)))


                # TEST NET
                elif strstr(line, 'Testing net'):
                    matched = match_iteration(line)
                    test_iteration.append(int(matched.group(1)) + base_test_iter)

                elif strstr(line, 'Test net output'):
                    matched = match_loss(line)
                    if matched:
                        test_loss.append(float(matched.group(1)))
                    else:
                        matched = match_accuracy(line)
                        test_accuracy.append(float(matched.group(1)))

    print("TRAIN", train_iteration, train_loss)
    print("TEST", test_iteration, test_loss)
    print("ACCURACY", test_iteration, test_accuracy)

    # loss
    plt.plot(train_iteration, train_loss, 'k', label='Train loss')
    plt.plot(test_iteration, test_loss, 'r', label='Test loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Number of iterations')
    plt.savefig('dlcrf_loss.jpg')

    # metrics - no CRF
    plt.clf()
    plt.plot(range(len(acc_nocrf)), acc_nocrf, 'k', label='accuracy (no CRF)')
    plt.plot(range(len(mean_recall_nocrf)), mean_recall_nocrf, 'r', label='avg. recall (no CRF)')
    plt.plot(range(len(mean_jaccard_nocrf)), mean_jaccard_nocrf, 'g', label='avg. Jaccard (no CRF)')
    plt.legend(loc=0)
    plt.savefig('dlcrf_metrics_noCRF.jpg')

    # metrics - with CRF
    plt.clf()
    plt.plot(range(len(acc_crf)), acc_crf, 'k', label='accuracy (CRF)')
    plt.plot(range(len(mean_recall_crf)), mean_recall_crf, 'r', label='avg. recall (CRF)')
    plt.plot(range(len(mean_jaccard_crf)), mean_jaccard_crf, 'g', label='avg. Jaccard (CRF)')
    plt.legend(loc=0)
    plt.savefig('dlcrf_metrics_CRF.jpg')


def strstr(a, b):
    return b in a


def match_iteration(line):
    return re.search(r'Iteration (.*),', line)


def match_loss(line):
    return re.search(r'loss = (.*)', line)


def match_accuracy(line):
    return re.search(r'seg-accuracy = (.*)', line)


def process_arguments(argv):
    if len(argv) < 2:
        help()

    log_files = argv[1:]
    return log_files


def help():
    print('Usage: python plot_loss_from_log_dlcrf.py [LOG_FILE]+\n'
          'LOG_FILE is text file containing log produced by caffe.'
          'At least one LOG_FILE has to be specified.'
          'Files has to be given in correct order (the oldest logs as the first ones).'
          , file=sys.stderr)

    exit()


if __name__ == '__main__':
    main()
