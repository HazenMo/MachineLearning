# Import all needed libraries.
from turtle import color
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pygame 
from PIL import Image
from PIL import Image, ImageOps

# Declare dataset.
mnist = tf.keras.datasets.mnist

# Pull relevant data from the set.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path = 'mnist.npz')

train_images = train_images / 255
test_images = test_images / 255

# Display dataset, to understand what the computer sees.
plt.figure(figsize=(8, 8))
for i in range (16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# Declare the "shape" of the model.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Run the model.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=7)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model to variable.
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
predictions[0]

np.argmax(predictions[0])
test_labels[0]

# Display model results to user.
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                    100*np.max(predictions_array),
                                    true_label),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 3
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Create a pygame window where the user can draw a new digit.
pygame.init()
white = [255,255,255]
black = [0,0,0]
size = [400,400]
poly_list = []
screen=pygame.display.set_mode(size)
pygame.display.set_caption("Test")
pygame.mouse.set_visible(100)
done = False
mouse_down = False
clock = pygame.time.Clock()

def draw_cursor(screen,x,y):
    if mouse_down:
        polygon = ([x,y], [x-22, y], [x-22,y-22], [x,y-22])
        poly_list.append(polygon)

while done==False:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=True
        """
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
        """
    screen.fill(white)
    pos = pygame.mouse.get_pos()
    mouse_down = pygame.mouse.get_pressed()[0]#note: returns 0/1, which == False/True
    x=pos[0]
    y=pos[1]
    draw_cursor(screen,x,y)
    for polygon in poly_list:
        pygame.draw.polygon(screen, black, polygon)
    pygame.display.flip()
    clock.tick(60)

pygame.image.save(screen, 'number.jpg')

pygame.quit()

# Save image from pygame and refigure to meet models needs.
image = Image.open('number.jpg').convert('L')
resized = image.resize((28, 28))
resized = ImageOps.invert(resized)


img = (np.expand_dims(resized, 0))

# Test digit against model.
predictions_single = probability_model(img)

# Get correct label from user.
test_input = int(input('What number did you draw? '))

# Display models predictions against correct label.
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(0, predictions_single[0], (test_input,), img)
plt.subplot(1,2,2)
plot_value_array(0, predictions_single[0],  (test_input,))
plt.show()