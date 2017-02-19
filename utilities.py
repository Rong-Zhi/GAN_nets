import os
from glob import glob
from scipy.ndimage import imread
import numpy as np
import shutil
from PIL import Image

import constant as c
def get_dir(directory):
    """
    If the given directory is not exist, create a new one.
    Otherwise, just return the given directory.

    :param directory: The path to the directory
    :return: The path to the directory
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory

def clear_dir(directory):
    """
    To remove all files in given directory

    :param directory: The path to the directory
    :return: N/A
    """
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

def get_test_dims(test_dir):
    """
    Read the frame size of the test frames.

    :param: The path to test videos
    :return: Height, Width
    """
    # read only on frame. aka the first frame
    img_path = glob(os.path.join(test_dir, '*/*'))[0]
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def get_train_dims(train_dir):
    """
    Read the frame size of the training frames

    :param train_dir: The path to train videos
    :return: Height, Width
    """
    img_path = glob(os.path.join(train_dir, '*/*'))[0]
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def get_test_frame_dims():
    img_path = glob(os.path.join(c.TEST_DIR, '*/*'))[0]
    img = imread(img_path, mode='RGB') #  to read a image, and return an array which representing the image
    shape = np.shape(img) # get the shape of the image, e.g. (210, 160)

    return shape[0], shape[1]   # return the row(210) and coloumn(160)

def get_train_frame_dims():
    img_path = glob(os.path.join(c.TRAIN_DIR, '*/*'))[0]
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]


def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames


def clip_l2_diff(clip):
    """
    @param clip: A numpy array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    @return: The sum of l2 differences between the frame pixels of each sequential pair of frames.
    """
    diff = 0
    for i in range(c.HIST_LEN):
        frame = clip[:, :, 3 * i:3 * (i + 1)]
        next_frame = clip[:, :, 3 * (i + 1):3 * (i + 2)]
        # noinspection PyTypeChecker
        diff += np.sum(np.square(next_frame - frame))

    return diff

def get_full_clips(data_dir, num_clips, num_rec_out=1):
    """
    Loads a batch of random clips from the unprocessed train or test data.

    @param data_dir: The directory of the data to read. Should be either c.TRAIN_DIR or c.TEST_DIR.
    @param num_clips: The number of clips to read.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape
             [num_clips, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    clips = np.empty([num_clips,
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH,
                      (3 * (c.HIST_LEN + num_rec_out))])

    # get num_clips random episodes
    ep_dirs = np.random.choice(glob(os.path.join(data_dir, '*')), num_clips)

    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num, ep_dir in enumerate(ep_dirs):
        ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
        clip_frame_paths = ep_frame_paths[start_index:start_index + (c.HIST_LEN + num_rec_out)]

        # read in frames
        for frame_num, frame_path in enumerate(clip_frame_paths):
            frame = imread(frame_path, mode='RGB')
            norm_frame = normalize_frames(frame)

            clips[clip_num, :, :, frame_num * 3:(frame_num + 1) * 3] = norm_frame

    return clips



def process_clip():
    """
    Gets a clip from the train dataset, cropped randomly to c.TRAIN_HEIGHT x c.TRAIN_WIDTH.

    @return: An array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
             A frame sequence with values normalized in range [-1, 1].
    """
    # Loads a batch of random clips from the unprocessed train or test data.
    clip = get_full_clips(data_dir=c.TRAIN_DIR, num_clips=1)[0]

    # Randomly crop the clip. With 0.05 probability, take the first crop offered, otherwise,
    # repeat until we have a clip with movement in it.
    take_first = np.random.choice(2, p=[0.95, 0.05])
    cropped_clip = np.empty([c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 3 * (c.HIST_LEN + 1)])
    for i in range(100):  # cap at 100 trials in case the clip has no movement anywhere
        crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
        cropped_clip = clip[crop_y:crop_y + c.TRAIN_HEIGHT, crop_x:crop_x + c.TRAIN_WIDTH, :]

        if take_first or clip_l2_diff(cropped_clip) > c.MOVEMENT_THRESHOLD:
            break

    return cropped_clip




def get_train_batch():
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """
    clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))],
                     dtype=np.float32)
    for i in range(c.BATCH_SIZE):
        path = c.TRAIN_DIR_CLIPS + str(np.random.choice(c.NUM_CLIPS)) + '.npz'
        clip = np.load(path)['arr_0']

        clips[i] = clip

    return clips


def train_img_save(img_preds, img_gts, batch_step):
    """

    :param img_preds:
    :param img_gts:
    :param batch_step:
    :return:
    """
    img_save_dirs = get_dir(c.IMG_SAVE_DIR + 'Step' + str(batch_step) + '/')
    for i in range(c.BATCH_SIZE):
        img_save_dir = get_dir(img_save_dirs + str(i) + '/')
        image = Image.fromarray(denormalize_frames(img_preds[3][i, :, :, :]), mode='RGB')
        image.save(img_save_dir + 'preds_03' + '.png')
        image = Image.fromarray(denormalize_frames(img_gts[3][i,:,:,:]), mode='RGB')
        image.save(img_save_dir + 'groundts_03'+'.png')
    #TODO: Also save the ground truth image. img_gts.
