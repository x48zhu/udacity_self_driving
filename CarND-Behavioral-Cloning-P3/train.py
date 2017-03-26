import csv
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


dir_root = '/Users/xi/Downloads/data/behavior_clone'
samples = []
with open(os.path.join(dir_root, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


def generator(samples, batch_size=32, is_train=False):
    num_samples = len(samples)
    if is_train:
        batch_size //= 2
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = os.path.join(dir_root, batch_sample[0].strip())
                left_name = os.path.join(dir_root, batch_sample[1].strip())
                right_name = os.path.join(dir_root, batch_sample[2].strip())

                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)

                correction = 0.1
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

                if is_train:
                    center_image = np.fliplr(center_image)
                    center_angle = - center_angle
                    left_image = np.fliplr(left_image)
                    left_angle = - left_angle
                    right_image = np.fliplr(right_image)
                    right_angle = - right_angle

                    images.append(center_image)
                    images.append(left_image)
                    images.append(right_image)
                    angles.append(center_angle)
                    angles.append(left_angle)
                    angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


train_samples, test_samples = train_test_split(samples, test_size=0.2)
train_samples, validation_samples = train_test_split(train_samples,
                                                     test_size=0.2)

train_generator = generator(train_samples, batch_size=32, is_train=True)
validation_generator = generator(validation_samples, batch_size=32)
test_generator = generator(test_samples, batch_size=32)

ch, row, col = 3, 80, 320

from model import model as model

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=1, verbose=1)

metrics = model.evaluate_generator(test_generator,
                                   val_samples=len(test_samples))

for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

print(history_object.history.keys())

# plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model.h5')

