# Gender Classification Using CNNs

I predicted gender using images extracted from social media profiles, resulting in an increase in accuracy from 0.59 to 0.71. This model creates improved user profiling based on gender. 

## Getting Started

Download [Tensorflow](https://www.tensorflow.org/) & Keras.

## Dataset
9,500 pictures were provided in the dataset, to ensure uniqueness each image was named with the convention: ‘userid’.jpg, and only allowed one image per userid. Additionally the dataset contained a csv file with a user's id and their gender. All data was anonymized prior to being recieved.

## Approach
Images were used to infer a social media user's gender. These images were collected from users who agreed to share their information for research purposes. The process for using images to generate gender predictions was threefold: (1) Pre-process images and label them so that they can be used as input into a supervised Convolutional Neural Network (CNN) (2) Train model using labeled images (3) Load trained model into deployment environment and test on unseen validation set.

### 1. Literature Review
The social media industry is in an unprecedented position to gather great amounts of user data. As of Q4 2017 Facebook alone registered 2.2 billion of active users monthly. As a result, businesses have adapted their marketing strategies moving away from generic magazine inserts and in favor of customized ads. This new form of advertisement, known as ‘user profiling’  is  targeted at each user based on data collected from their social media. User profiling is a growing industry, to the point that social media advertising budgets have grown drastically from $16 billion in 2014 to $31 billion in 2016. 
In this project, we explore the capabilities of supervised machine learning to determine valuable information about social media users, specifically their gender. 

### 2. Data Pre-processing
The original image dataset contained only user profile images labeled with their userid. Unfortunately, this information was insufficient for training our CNN, which required a directory of images per classification as input. To generate a training and validation dataset a database containing a relationship between the user identification number (userid) and his/her gender was used. Once the labeling was completed, the photos associated with female/male userid were moved into their appropriate train and test folders. A train/test split of 8500 train/1000 test was employed. The validation set contained images from 333 users. 

### 3. Training
A CNN is a multilayer neural network that begins with one or more convolutional layers and then contains fully connected layers. Convolutional Neural Networks (CNN) are well suited for answering the question “What gender is this user based on their profile picture?” because the architecture of a CNN is built for the 2D inputs, perfectly mirroring the structure of images. 
A sequential model was built with three layers of 2D CNNs using Keras and TensorFlow. The first two CNN's each had a 32-dimensional output space and Relu activation. The final CNN contained a 64-dimensional output space and Sigmoid activation. Sigmoid activation was chosen for the final CNN layer because sigmoid functions exists between 0 and 1, as opposed to ReLU which exists between 0 and ∞. Since the model is predicting the probability (from 0 to 1) that each image will fall into class female or class male, sigmoid was the natural choice.
Images were then read in from the validation and training directories and the model was trained. 
Because of the small size of the training dataset, overfitting was a concern. Overfitting happens when your model is trained on a subset of data too small to closely approximate the entire dataset. The results of the efforts to prevent overfitting will be discussed in the Results section below.The following techniques were used to reduce the risk of overfitting:
* Using TensorFlow’s built in .flow_from_directory to generate augmented batches of images for the training and testing set. This prevents the model from ever being trained on the same image more than once.
* Keras’ MaxPooling2D function with pool_size=(2,2) to halve input size in both dimensions. Not only does this combat overfitting by generating an abstracted representation of the image but it reduces our computational time (which is valuable when training a neural network using images on a Macbook Air :-)).
* Dropout(0.25) randomly set 0.25x input units to 0 which prevents co-adaptations on training data and requires the CNN to learn more robust features. This can be useful when applied in conjunction with random subsets of other neurons. 

### 4. Training 
Once the model was trained, learned weights were saved into a .h5 file. This .h5 file was later loaded into a new model that generated predictions on the validation set.  

### 5. Results
For the task of gender classification a deployment accuracy of 0.69 was achieved using a 2D CNN. This is well above the baseline results of 0.59 accuracy.
Initially overfitting the model was a large concern, so much so that the initial CNN had four dropout layers, and three MaxPooling2D layers. With this architecture it was a struggle to pass the baseline of 0.59 for gender classification, squeezing by with a 0.62 (local) validational accuracy. Once deployed on the VM this same model achieved an accuracy of 0.69, which lead me to believe that with this architecture it was likely that I was underfitting the data. Once the number of dropout and MaxPooling2D layers were reduced, it was possible to achieve accuracy in the low to mid 70s (%).
I spent a lot of time experimenting with batch_size and number of epochs. One epoch accounts for a single forward and backward pass through the neural network, since this is usually too computationally intensive on a standard computer we divide this into smaller batches. The batch size determines the training examples contained in a single batch. I was not surprised to see lower accuracy when the batch_size was decreased (and epochs remained unchanged) because with small batch sizes the gradients learned are only rough approximations of the true gradients, so it takes more epochs to converge. For our model the best results were obtained with 25 epochs.

## Acknowledgements
Thank you to Raisa Thomas & Binh Hua for their support during this project!
