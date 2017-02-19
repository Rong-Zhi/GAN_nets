from generator_model import Generator_class
from keras.models import Sequential
import utilities as u
import constant as c
from PIL import Image
class GANsRunner:

    def __init__(self, num_steps, num_test_rec):
        """
        :param num_steps: The number of training steps to run.
        :param num_test_rec: The number of recursive generations to produce when testing.
        """
        self.global_step = 0
        self.num_steps = num_steps
        self.num_test_rec = num_test_rec
        self.model_4x4 = Sequential()
        self.model_8x8 = Sequential()
        self.model_16x16 = Sequential()
        self.model_32x32 = Sequential()


        print("Init generator...")
        self.generators = Generator_class(models=[self.model_4x4,
                                                  self.model_8x8,
                                                  self.model_16x16,
                                                  self.model_32x32])

    def train(self):
        """
        Runs a training loop on the model networks
        :return:
        """
        for batch_step in range(100):
            print("Batch Step: {}".format(batch_step))
            batch_preds, batchs_gts = self.generators.train_batch(batch=u.get_train_batch())
            if batch_step % c.IMG_SAVE_FREQ == 0:
                print("Step: {0}, Saving Images...".format(batch_step))
                u.train_img_save(img_preds=batch_preds, img_gts=batchs_gts,batch_step=batch_step)
            #TODO: the training speed is extremely slow. Dont know why yet.

    def test(self):
        """
        :return:
        """
        self.global_step = self.generators.test_train_batch(batch=u.get_train_batch())
        return True



def main():
    # Self-define-parameters
    test_only = False
    num_test_rec = 1
    num_steps = 1

    #
    runner = GANsRunner(num_steps, num_test_rec)
    runner.train()

# this only executes when file is executed rather than import.
if __name__ == '__main__':
    main()
