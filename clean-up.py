import pandas as pd
import matplotlib.pyplot as plt
import math
import random

from PIL import Image

# Import Styles CSV
styles_df = pd.read_csv("M:\\fashion-dataset\styles.csv")

# Create & Show All Styles
style_map = {}
for i, style in enumerate(styles_df['subCategory'].unique()):
	style_map[style] = i

print('Styles:\n{}'.format(style_map))

# Create & Show All Bad Styles
bad_styles = [
	'Socks',
	'Innerwear',
	'Shoe Accessories',
	'Fragrance',
	'Lips',
	'Saree',
	'Apparel Set',
	'Mufflers',
	'Skin Care',
	'Makeup',
	'Free Gifts',
	'Accessories',
	'Skin',
	'Beauty Accessories',
	'Water Bottle',
	'Eyes',
	'Bath and Body',
	'Sports Accessories',
	'Sports Equipment',
	'Stoles',
	'Hair',
	'Perfumes',
	'Home Furnishing',
	'Umbrellas',
	'Vouchers',
	'Wristbands',
	'Wallets',
	'Nails',
	'Cufflinks'
]
bad_images = {}

for index, row in styles_df.iterrows():
	if row['subCategory'] in bad_styles:
		filepath = "M:\\fashion-dataset\\images\\" + str(row['id']) + '.jpg'
		if row['subCategory'] in bad_images:
			bad_images[row['subCategory']].append(filepath)
		else:
			bad_images[row['subCategory']] = [filepath]

for k, v in bad_images.items():
	print('{}: {} images'.format(k, len(v)))
'''
# Show the Bad Images
# Set the number of rows and columns for the grid
sqrt_n = math.sqrt(len(bad_styles))
num_rows = math.ceil(sqrt_n)
num_cols = math.ceil(len(bad_styles)/sqrt_n)

# Create a figure object and set the size
fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# Loop over the image filenames and display each image in a subplot
i = 0
for k, v in bad_images.items():
    # Load the image from file

    img_path = random.choice(v)
    img = Image.open(img_path)

    # Calculate the subplot index based on the row and column
    row = i // num_cols
    col = i % num_cols
    i += 1

    # Display the image in the subplot
    axs[row, col].imshow(img)
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('{}({})'.format(k, len(v)))

# Adjust the spacing between the subplots
plt.tight_layout()

# Display the figure
plt.show()

'''

# Removing Bad Images
new_styles = styles_df[~styles_df['subCategory'].isin(bad_styles)].dropna()
new_styles.to_csv("M:\\fashion-dataset\\new_styles.csv")

new_style_map = {}
for i, style in enumerate(new_styles['subCategory'].unique()):
	new_style_map[style] = i

print('Styles:\n{}'.format(new_style_map))
