import random
import numpy as np
import matplotlib.pyplot as plt

# Randomly select an index from the test dataset
idx = random.randrange(len(X_test))

# Fetch the test image at the selected index
img = X_test[idx]

# Expand dimensions to simulate a batch of size 1 and make prediction
pred = model.predict(img[None], verbose=0)

# Determine the predicted class label
label = labels[np.argmax(pred)]

# Display the image with the predicted label
plt.imshow(img.squeeze(), cmap='gray' if img.ndim == 3 and img.shape[-1] == 1 else None)
plt.title(f"Predicted: {label}")
plt.axis('off')
plt.show()
