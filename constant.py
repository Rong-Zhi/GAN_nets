import os
from glob import glob
import utilities as u

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # <1> data
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# Root directory for all data
DATA_DIR = u.get_dir('./Data/')
# Directory of unprocessed training videos/frames
TRAIN_DIR = u.get_dir(os.path.join(DATA_DIR, 'Train/'))
# Directory of unprocessed test videos/frames
TEST_DIR = u.get_dir(os.path.join(DATA_DIR, 'Test/'))
# Directory of processed training clips
TRAIN_DIR_CLIPS = u.get_dir(os.path.join(DATA_DIR, 'Clips/'))

# For processing clips, l2 diff between frames must greater than this
MOVEMENT_THRESHOLD = 100
# total number of processed clips in TRAIN_DIR_CLIPS
NUM_CLIPS = len(glob(TRAIN_DIR_CLIPS + '*'))

# the height and width of the full frames to test on. Set in avg_runner.py or process_data.py main.
FULL_HEIGHT, FULL_WIDTH  = u.get_train_frame_dims()
# the height and width of the patches to train on
TRAIN_HEIGHT = 32
TRAIN_WIDTH = 32

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # <2> Output
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# root directory for all saved content
SAVE_DIR = u.get_dir('./Save/')
# directory for saved models
MODEL_SAVE_DIR = u.get_dir(os.path.join(SAVE_DIR, 'Models/'))
# directory for saved TensorBoard summaries
SUMMARY_SAVE_DIR = u.get_dir(os.path.join(SAVE_DIR, 'Summaries/'))
# directory for saved images
IMG_SAVE_DIR = u.get_dir(os.path.join(SAVE_DIR, 'Images/'))

# how often to print loss/train error stats, in # steps
STATS_FREQ      = 10
# how often to save the summaries, in # steps
SUMMARY_FREQ    = 100
# how often to save generated images, in # steps
IMG_SAVE_FREQ   = 5
# how often to test the model on test data, in # steps
TEST_FREQ       = 5000
# how often to save the model, in # steps
MODEL_SAVE_FREQ = 10000




# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # #  <3> Loss parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# for lp loss. e.g, 1 or 2 for l1 and l2 loss, respectively)
L_NUM = 2
# the power to which each gradient term is raised in GDL loss
ALPHA_NUM = 1
# the percentage of the adversarial loss to use in the combined loss
LAM_ADV = 0.05
# the percentage of the lp loss to use in the combined loss
LAM_LP = 1
# the percentage of the GDL loss to use in the combined loss
LAM_GDL = 1


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # #  <4> General training
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# whether to use adversarial training vs. basic training of the generator
ADVERSARIAL = True
# the training minibatch size
BATCH_SIZE = 8
# the number of history frames to give as input to the network
HIST_LEN = 4
# Color model is RGB
RGB_SIZE = 3
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # #  <5> Generator model
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# learning rate for the generator model
LRATE_G = 0.00004  # Value in paper is 0.04
# padding for convolutions in the generator model
PADDING_G = 'SAME'
# feature maps for each convolution of each scale network in the generator model
# e.g SCALE_FMS_G[1][2] is the input of the 3rd convolution in the 2nd scale network.
# SCALE_FMS_G = [[3 * HIST_LEN, 128, 256, 128, 3],                        # scale 1   (4 x 4)
#                [3 * (HIST_LEN + 1), 128, 256, 128, 3],                  # scale 2   (8 x 8)
#                [3 * (HIST_LEN + 1), 128, 256, 512, 256, 128, 3],        # scale 3   (16x16)
#                [3 * (HIST_LEN + 1), 128, 256, 512, 256, 128, 3]]        # scale 4   (32x32)
SCALE_FMS_G = [[3* HIST_LEN, 128, 256, 128, 3],                        # scale 1   (4 x 4)
               [3*(HIST_LEN + 1), 128, 256, 128, 3],                  # scale 2   (8 x 8)
               [3*(HIST_LEN + 1), 128, 256, 512, 256, 128, 3],        # scale 3   (16x16)
               [3*(HIST_LEN + 1), 128, 256, 512, 256, 128, 3]]        # scale 4   (32x32)
# kernel sizes for each convolution of each scale network in the generator model
SCALE_KERNEL_SIZES_G = [[3, 3, 3, 3],           # scale 1   (4 x 4)
                        [5, 3, 3, 5],           # scale 2   (8 x 8)
                        [5, 3, 3, 3, 3, 5],     # scale 3   (16x16)
                        [7, 5, 5, 5, 5, 7]]     # scale 4   (32x32)
#
SCALE_SIZE = [4, 8, 16, 32]

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # #  <6> Discriminator model
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# learning rate for the discriminator model
LRATE_D = 0.02
# padding for convolutions in the discriminator model
PADDING_D = 'VALID' # no padding.
# feature maps for each convolution of each scale network in the discriminator model
SCALE_CONV_FMS_D = [[3, 64],
                    [3, 64, 128, 128],
                    [3, 128, 256, 256],
                    [3, 128, 256, 512, 128]]
# kernel sizes for each convolution of each scale network in the discriminator model
SCALE_KERNEL_SIZES_D = [[3],
                        [3, 3, 3],
                        [5, 5, 5],
                        [7, 7, 5, 5]]
# layer sizes for each fully-connected layer of each scale network in the discriminator model
# layer connecting conv to fully-connected is dynamically generated when creating the model
SCALE_FC_LAYER_SIZES_D = [[512, 256, 1],
                          [1024, 512, 1],
                          [1024, 512, 1],
                          [1024, 512, 1]]