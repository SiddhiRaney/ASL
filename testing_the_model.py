import random
import numpy as np
import matplotlib.pyplot as plt

# Select a random test image
idx = random.randint(0, len(X_test) - 1)
img = X_test[idx]

# Predict the label
pred = model.predict(img[np.newaxis, ...], verbose=0)
predicted_label = labels[np.argmax(pred)]

# Display the image with the predicted label
plt.imshow(img.squeeze(), cmap='gray' if img.ndim == 3 and img.shape[-1] == 1 else None)
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
