from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import NASNetLarge
import os
import matplotlib.pyplot as plot
import numpy as np
from sklearn.metrics import confusion_matrix

def main():
    # ################### DATA #############################
    # In de script maken we gebruik van de generators
    # Data augementation word aan en uigezet door regels van code te uncommenten
    epoch = 30


    base_dir = 'data'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    validation_dir = os.path.join(base_dir, 'validation')

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       # rotation_range=40,
                                       # width_shift_range=0.2,
                                       # height_shift_range=0.2,
                                       # shear_range=0.2,
                                       # zoom_range=0.2,
                                       # horizontal_flip=True
                                       ) # recaling images

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(150,150),
                                                        shuffle=True,
                                                        batch_size=32,
                                                        # save_to_dir='train_test',
                                                        class_mode='categorical')

    for data_batch, labels_batch in train_generator:
        print('\ndata_generator_flow batch shape: ', data_batch.shape)
        print('\nlabel batch shape: ', labels_batch.shape)
        break

    # validation data_generator_flow should not augmented
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  target_size=(150,150),
                                                                  batch_size=20,
                                                                  class_mode='categorical')

    # ################### MODEL ############################
    print('\n==================================================================')
    print('\nPRETRAINED MODEL + FEATURE EXTRACTION')
    print('\nMAKING AND COMPILING THE MODEL.\n')

    conv_base = VGG16(weights='imagenet',  # which checkpoints to initialize from
                      include_top=False,
                      # refers to wether or not the densely connected classifier in top of the network should be included
                      input_shape=(150, 150, 3))
    print('Summary of pretrained network:\n')
    conv_base.summary()

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    # Before you compile and train the model, it’s very important to freeze the convolutional base.
    # Freezing a layer or set of layers means preventing their weights from being updated during training.
    # If you don’t do this, then the representations that were previously learned by the convolutional base
    # will be modified during training.

    print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))
    conv_base.trainable = False
    print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))

    model.compile(optimizer=optimizers.RMSprop(2e-5),
                  loss='categorical_crossentropy',
                  metrics='accuracy')

    history = model.fit(train_generator,
                        epochs=epoch,
                        validation_data=validation_generator,
                        verbose=1)

    #model.save('borden_covnet_1_no_opt.h5')

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss= history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, 1 + len(accuracy))
    plot.plot(epochs, accuracy, 'bo', label='Training Accuracy')
    plot.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plot.title('Training and Validation accuracy')
    plot.legend()
    plot.figure()
    plot.plot(epochs, loss, 'bo', label="Training Loss")
    plot.plot(epochs, val_loss, 'b', label="Validation Loss")
    plot.title('Training and Validation loss')
    plot.legend()
    plot.figure()
    plot.show()

    # ################### TESTING ##########################
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(150,150),
                                                      class_mode='categorical')

    for test_images, test_labels in test_generator:
        test_images = test_images
        test_labels = test_labels
        break

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
