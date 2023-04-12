import io
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from urllib.request import urlopen
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Softmax, Dropout

# Load the styles.csv file
styles_df = pd.read_csv('M:/fashion-dataset/new_styles.csv')
images_df = pd.read_csv('M:/fashion-dataset/images.csv')

# Create a dictionary to map style names to integer labels
style_map = {}
r_style_map = {}
for i, style in enumerate(styles_df['subCategory'].unique()):
	r_style_map[i] = style
	style_map[style] = i

# Load the images and their labels
X = []
y = []

image_amount = 200  # @param {type:"slider", min:1, max:10000, step:1}
time_limit = 5  # @param {type:"integer"}

# @markdown Time limit is in minutes
time_limit *= 60
timer = 0
start_time = time.time()
for index, row in styles_df.iterrows():
	timer = time.time() - start_time
	print('Time: %.2f/%d' % (timer, time_limit), end='\r')

	# uses set amount of images to fit
	if index == image_amount or timer >= time_limit:
		break

	img_path = 'M:/fashion-dataset/images/' + str(row['id']) + '.jpg'

	if os.path.isfile(img_path):
		print('Image Found', end='\r')
		# Create a PIL image object from the image data
		pil_image = Image.open(img_path)
	else:
		print('Downloading Image...', end='\r')
		try:
			# grabs the image download url
			img_path = images_df.loc[images_df['filename'] == str(row['id']) + '.jpg']
			img_path = img_path.iloc[0].loc['link']

			with urlopen(img_path) as url_response:
				image_data = url_response.read()

			# Create a PIL image object from the image data
			pil_image = Image.open(io.BytesIO(image_data))

		except():
			print('Error Downloading Image')
			continue

	# Resize the image
	pil_image = pil_image.resize((224, 224))

	# Convert the PIL image to a numpy array
	np_image = np.asarray(pil_image)

	X.append(np_image)
	y.append(style_map[row['subCategory']])

X = np.array(X)
y = np.array(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)

# Normalize the pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0

print('Time: %.1f \t Images Downloaded: %d/%d' % (timer, len(X), styles_df.shape[0]))

# Define the model architecture
model = Sequential([
	# Add convolutional layer
	Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
	Dropout(0.05),
	# Add max pooling layer
	MaxPooling2D(2, 2),
	# Add another convolutional layer
	Conv2D(32, (3, 3), activation='relu'),
	Dropout(0.05),
	# Add another max pooling layer
	MaxPooling2D(2, 2),
	# Flatten the output from convolutional layers
	Flatten(),
	# Add a fully connected layer with 64 neurons
	Dense(64, activation='relu'),
	Dropout(0.05),
	# Add an output layer with softmax activation
	Dense(len(style_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit Model & Predict
model.fit(X_train, y_train, epochs=7)
predictions = model(X_train).numpy()


def normalize(predictions_array):
	new_array = []
	for prediction in predictions_array:
		new_array.append(prediction / max(prediction))
	return np.asarray(new_array)


def plot_image(i, predictions_array, true_label, img):
	true_label, img = true_label[i], img[i]

	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array) % len(style_map)
	true_label %= len(style_map)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	for key, value in style_map.items():
		if value == predicted_label:
			predicted_label = key
		if value == true_label:
			true_label = key

	plt.xlabel("{} {:2.0f}% ({})".format(predicted_label, 100 * np.max(predictions_array), true_label), color=color)


def plot_value_array(i, predictions_array, true_label, top_n=3):
	# Get the top_n predicted labels and their corresponding probabilities
	sorted_idxs = np.argsort(predictions_array)[::-1]
	top_n_idxs = sorted_idxs[:top_n]
	top_n_probs = predictions_array[top_n_idxs]

	true_label = true_label[i]
	plt.grid(False)
	plt.xticks(range(top_n), top_n_idxs)
	plt.yticks([])

	# Plot the top_n predicted labels and their probabilities
	thisplot = plt.bar(top_n_idxs, top_n_probs, color="#777777")
	plt.ylim([0, 1])

	# Set the color of the bars corresponding to the top_n predicted labels
	for j in range(top_n):
		if top_n_idxs[j] != true_label:
			thisplot[j].set_color('red')
		else:
			thisplot[j].set_color('blue')


def plot_predictions(i, predictions_array, true_label, top_n=3):
	sorted_idxs = np.argsort(predictions_array)[::-1]
	top_n_idxs = sorted_idxs[:top_n]
	top_n_probs = predictions_array[top_n_idxs]
	true_label = true_label[i]

	# Show the top_n predicted labels and their probabilities
	for j in range(top_n):
		label = f"{r_style_map[top_n_idxs[j]]} ({100 * top_n_probs[j]:.2f}%)"
		plt.text(1, j, label, fontsize=6, color='blue' if top_n_idxs[j] == true_label else 'red', ha='center', va='center')

	plt.xticks([])
	plt.yticks(range(top_n))
	plt.xlim([-0.5, top_n - 0.5])
	plt.ylim([-1, top_n])


# predictions = normalize(predictions)

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
	plot_image(i, predictions[i], y_val, X_val)
	plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
	plot_predictions(i, predictions[i], y_val)
plt.tight_layout()
plt.show()
