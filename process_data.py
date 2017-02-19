import numpy as np
import getopt
import sys
from glob import glob
import os

import constant as c
import utilities as u


def process_training_data(num_clips):
    """
    Processes random training clips from the full training data. Saves to TRAIN_DIR_CLIPS by
    default.

    @param num_clips: The number of clips to process. Default = 5000000 (set in __main__).

    @warning: This can take a couple of hours to complete with large numbers of clips.
    """
    num_prev_clips = len(glob(c.TRAIN_DIR_CLIPS + '*'))

    for clip_num in range(num_prev_clips, num_clips + num_prev_clips):
        clip = u.process_clip()

        np.savez_compressed(c.TRAIN_DIR_CLIPS + str(clip_num), clip)

        if (clip_num + 1) % 100 == 0:
            print('Processed %d clips' % (clip_num + 1))

def main():

    num_clips = 2000

    # set train frame dimensions
    assert os.path.exists(c.TRAIN_DIR)
    c.FULL_HEIGHT, c.FULL_WIDTH = u.get_train_frame_dims()

    ##
    # Process data for training
    ##

    process_training_data(num_clips)


if __name__ == '__main__':
    main()
