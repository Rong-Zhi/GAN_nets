# GAN_nets
This is the Keras version of video prediction.

generator_model.py is Keras version of generator model, with Tensorflow backend.

g_model.py is Tensorflow version of generator model.

Now, I want to modify the network to be suitable for an intuitive physics learning. To do that, I need two views with the similar viewpoint(but not identical) theoretically. We want to train those two videos simultaneously to get the dense depth map.
