from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import matplotlib.pyplot as plot
import numpy as np
from sklearn.metrics import confusion_matrix

def main():
    # ################### DATA #############################
    # In de script maken we gebruik van APARTE generators voor de 3 sets
    # Data augementation word aan en uigezet door regels van code te uncommenten

    # we hebben twee folders met data die we kunnen gebruiken als folder voor input voor de flow_from directory functie.

    # folder "data_generator_flow" bevat augmented images die gegeneerd zijn door train_datagen.flow() functie.
    # Deze folder word dan als input parameter meegegeven aan de flow_from_directory fcuntie

    # we hebben ook folder "data" die de 3 sets bevat, train, validation en test sets, waarbij de originele afbeeldingen verspreid zijn over de 3 sets door de main script
    epoch = 30
    batch_size=32
    lr = 1e-4
    loss = 'categorical_crossentropy'#'kullback_leibler_divergence'#
    base_dir = 'data'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    validation_dir = os.path.join(base_dir, 'validation')

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True
                                       ) # recaling images

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(150,150),
                                                        color_mode='rgb',
                                                        shuffle=True,
                                                        # save_to_dir='save_dir',
                                                        batch_size=batch_size,
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
                                                                  shuffle=False,
                                                                  color_mode='rgb',
                                                                  class_mode='categorical')
    # ################### MODEL ############################
    print('\n==================================================================')
    print('\nNO OPTIMALISATIONS')
    print('\nMAKING AND COMPILING THE MODEL.\n')
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())                 # Last Dense layer needs 1 vector.
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    print(model.summary())

    model.compile(loss=loss,
                  optimizer=optimizers.RMSprop(lr),
                  metrics=['accuracy'])

    print('\n==================================================================')
    print('FITTING AND SAVING THE MODEL.')

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
