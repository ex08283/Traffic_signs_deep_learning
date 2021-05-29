from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import layers

from tensorflow.keras.applications import VGG16
import numpy as np
import os
import matplotlib.pyplot as plot
from tensorflow.keras.preprocessing import image_dataset_from_directory

from sklearn.metrics import confusion_matrix

def main():
    epoch =30
    # ################### DATA #############################
    # In deze script generaren we batches van images, door gebruik te maken de generators en daarmee batches te genereren,
    # die we appenden naar een arraay, wat als 1 grote datasset word gezien.

    # Data augementation word aan en uigezet door regels van code te uncommenten
    base_dir = 'data_generator_flow'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # ################### PREPROCESSING ####################

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(150, 150),
                                                        # All images will be resized to 150x150.
                                                        batch_size=20,
                                                        # Since we use binary_crossentropy loss, we need binary labels.
                                                        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            target_size=(150, 150),
                                                            batch_size=29,
                                                            class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(validation_dir,
                                                      target_size=(150, 150),
                                                      batch_size=29,
                                                      class_mode='categorical')

    train_images = np.empty((0, 150, 150, 3), 'float32')
    train_labels = np.empty((0, 5), 'float32')
    i = 0
    for train_batch, trainlabels_batch in train_generator:
        i = i + 1
        train_images = np.append(train_images, train_batch, axis=0)
        train_labels = np.append(train_labels, trainlabels_batch, axis=0)
        if i == 150: break

    for validation_batch, validationlabels_batch in validation_generator:
        validation_images = validation_batch
        validation_labels = validationlabels_batch
        break

    for test_batch, testlabels_batch in test_generator:
        test_images = test_batch
        test_labels = testlabels_batch
        break

    # ################### MODEL ############################

    print('\n==================================================================')
    print('\nPRETRAINED MODEL + FEATURE EXTRACTION')
    print('\nMAKING AND COMPILING THE MODEL.\n')

    conv_base = VGG16(weights='imagenet',  # which checkpoints to initialize from
                      include_top=False,
                      # refers to wether or not the densely connected classifier in top of the network should be included
                      input_shape=(150, 150, 3))
    conv_base.trainable = False
    print('Summary of pretrained network:\n')
    conv_base.summary()

    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(2e-5),
                  loss='categorical_crossentropy',
                  metrics='accuracy')

    history = model.fit(train_images, train_labels, epochs=epoch,steps_per_epoch=100, batch_size=20, validation_data=(validation_images, validation_labels))

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(accuracy) +1 )

    plot.plot(epochs, accuracy, 'bo', label="Training Accuracy")
    plot.plot(epochs, val_accuracy, 'b', label="Validation Accuracy")
    plot.title('Training and validation accuracy epochs = %s' % epoch)
    plot.legend()
    plot.figure()
    plot.plot(epochs, loss, 'bo', label="Training Loss")
    plot.plot(epochs, val_loss, 'b', label="Validation Loss")
    plot.title('Training and validation loss epochs =%s' % epoch)
    plot.legend()
    plot.figure()
    plot.show()

    # ################### TESTING ##########################
    # You can generate the likelihood of reviews being POSITIVE by using the predict method:
    print('\nTesting test review 0.')
    predictions = model.predict(test_images)
    print('Test review 0 predicted as', predictions[0])

    print('Predictions:\n')

    for i in range(10):
        print("Target = %s, Predicted = %s" % (test_labels[i], predictions[i]))

    # Show the inputs (see line 38 and 39) and predicted outputs.

    print('\nEvaluating overall performance : [loss, accuracy].')
    results = model.evaluate(test_images, test_labels)
    print(results)


    def create_cm(test_labels, pred_labels):
        y_true = [np.argmax(i) for i in test_labels]
        y_pred = [np.argmax(i) for i in pred_labels]
        return confusion_matrix(y_true, y_pred)

    classes = ['Aanwijzingsborden', 'Gebodsborden', 'Gevaarsborden', 'Verbodsborden', 'Voorrangsborden']
    cm = create_cm(test_labels, predictions)

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plot.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plot.imshow(cm, interpolation='nearest', cmap=cmap)
        plot.title(title)
        plot.colorbar()
        tick_marks = np.arange(len(classes))
        plot.xticks(tick_marks, classes, rotation=45)
        plot.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plot.text(j, i, format(cm[i, j], fmt),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")

        plot.ylabel('True label')
        plot.xlabel('Predicted label')
        plot.tight_layout()
        plot.show()

    plot_confusion_matrix(cm, classes, title='Confusion matrix')


if __name__ == '__main__':
    main()
