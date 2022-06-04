"""
=====================
Example code: VoiceHD 
=====================

This is a simple example that runs VoiceHD[1] on OpenHD

VoiceHD: Imani, M., Kong, D., Rahimi, A., & Rosing, T. (2017, November). Voicehd: Hyperdimensional computing for efficient speech recognition. In 2017 IEEE international conference on rebooting computing (ICRC) (pp. 1-8). IEEE.
"""

import sys
import struct
import argparse

import numpy as np
import openhd as hd

# Use the below two lines to see the development logs
#from openhd.dev.debug import set_log_level, DEV
#set_log_level(DEV)

def validate(labels, pred_labels):
    n_correct = (pred_labels == labels).sum()
    n_labels = len(labels)
    print(n_correct, n_labels, n_correct / float(n_labels) * 100)

# 0. Command line argument parsing
def parse_argument():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
            '-t', '--training', 
            help='Training dataset filename (CHOIR DAT)',
            required=True)

    argparser.add_argument(
            '-i', '--inference',
            help='Inference dataset filename (CHOIR DAT)',
            required=True)

    args = argparser.parse_args()
    return args.training, args.inference

train_filename, test_filename = parse_argument()

# 1. Testing parameters ########################################################
Q = 10
D = 10000
hd.init(D=D, context=globals())

# Load or create dataset 
if train_filename:
    feature_matrix, labels, n_classes = hd.utils.read_choir_dat(train_filename)

    if not test_filename:
        feature_matrix = hd.utils.MatrixNormalizer().norm(feature_matrix)
        feature_matrix_tst = None
        labels_tst = None
    else:
        feature_matrix_tst, labels_tst, n_classes_tst = hd.utils.read_choir_dat(
                test_filename)
        assert(n_classes == n_classes_tst)
        feature_matrix, feature_matrix_tst = \
                hd.utils.MatrixNormalizer().norm_two(
                        feature_matrix, feature_matrix_tst)
        N_tst = feature_matrix_tst.shape[0]

    N = feature_matrix.shape[0]
    F = feature_matrix.shape[1]
    print(N,F,N_tst)
else:
    # Randomly generated features
    F = 500
    N = 20000
    feature_matrix = np.random.uniform(0., 1., (N, F)).astype(np.float32)
    labels = None
    nClasses = None
    feature_matrix_tst = None
################################################################################


# 2. Create base hyprevectors ##################################################
@hd.run
def create_random_bases():
    id_base = hd.draw_random_hypervector()
    level_base = hd.draw_random_hypervector()
    return id_base, level_base

@hd.run
def create_ids(F, id_base):
    id_hvs = hd.hypermatrix(F) # np.zeros(F, N) (not the empty list) 
    for f in range(F):
        id_hvs[f] = hd.permute(id_base, f)

    return id_hvs

@hd.run
def create_levels(Q, level_base):
    level_hvs = hd.hypermatrix(Q+1) # np.zeros((Q+1), N) (not the empty list)
    for q in range(Q+1):
        idx = int(q/float(Q) * D) / 2
        level_hvs[q] = hd.flip(level_base, idx)
        level_hvs[q] = hd.shuffle(level_hvs[q], 0)

    return level_hvs

with hd.utils.timing("Base hypervectors"):
    id_base, level_base = create_random_bases()
    id_hvs = create_ids(F, id_base)
    level_hvs = create_levels(Q, level_base)

################################################################################

# 3. Encode ####################################################################
def preprocesser(
        org_feature, cnv_feature, # Predefined argument (single feature)
        Q, level_hvs, id_hvs): # arguments passed by args
    cnv_feature = int(org_feature * Q)

def encoder(
        input_features, output_hypervector, # Predefined arguments
        Q, level_hvs, id_hvs): # arguments passed by args
    for f in range(F):
        output_hypervector += level_hvs[input_features[f]] * id_hvs[f]

    # This will be converted:
    #__preproc_f__ = 0
    #if __preproc_f__:
    #    __shared_features__[__preproc_f__] = int(input_features[__preproc_f__] * Q)

    #for f in range(F):
    #    output_hypervector += level_hvs[__shared_features__[f]] * id_hvs[f]


with hd.utils.timing("Encode training"):
    hv_matrix = hd.encode(
            encoder, extra_args = (Q, level_hvs, id_hvs),
            feature_matrix = feature_matrix,
            preprocess_function = preprocesser # optional
            )

if feature_matrix_tst is not None:
    with hd.utils.timing("Encode testing"):
        hv_matrix_tst = hd.encode(
                encoder, extra_args = (Q, level_hvs, id_hvs),
                feature_matrix = feature_matrix_tst,
                preprocess_function = preprocesser # optional
                )

################################################################################


# 4. Single-pass learning ######################################################
if labels is None:
    print("The program exits because no training dataset is given.")
    sys.exit()

@hd.run
def single_pass(hv_matrix, labels, N, n_classes):
    class_hvs = hd.hypermatrix(n_classes)

    for idx in range(N):
        class_hvs[labels[idx]] += hv_matrix[idx]

    return class_hvs

with hd.utils.timing("Single pass"):
    class_hvs = single_pass(hv_matrix, labels, N, n_classes)
    class_hvs.debug_print_values()
################################################################################


# 4-1. Retraining ##############################################################
@hd.run
def retrain(class_hvs, hv_matrix, labels, N, n_classes):
    search_results = hd.search(class_hvs, hv_matrix)

    for idx in range(N):
        if search_results[idx] != labels[idx]:
            class_hvs[labels[idx]] += hv_matrix[idx]
            class_hvs[search_results[idx]] -= hv_matrix[idx]

    return class_hvs

#RETRAIN_ITERATIONS = 0 # If not need to retrain
RETRAIN_ITERATIONS = 10
SHOW_STEP_RESULT = False
for it in range(RETRAIN_ITERATIONS):
    with hd.utils.timing("Retrain itereation: %d" % it):
        class_hvs = retrain(class_hvs, hv_matrix, labels, N, n_classes)

    if SHOW_STEP_RESULT and labels_tst is not None:
        validate(labels_tst, hd.search(class_hvs, hv_matrix_tst).to_numpy())
################################################################################


# 5. Testing ###################################################################
if labels_tst is None:
    print("The program exits because no testing dataset is given.")
    sys.exit()

@hd.run
def assoc_search(class_hvs, hv_matrix_tst):
    ret = hd.search(class_hvs, hv_matrix_tst)
    return ret

with hd.utils.timing("Testing with class model"):
    search_results = assoc_search(class_hvs, hv_matrix_tst)

validate(labels_tst, search_results.to_numpy())
################################################################################

