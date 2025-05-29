import random
import numpy as np
import matplotlib.pyplot as plt

# Select a random test image from the test dataset
idx = random.randint(0, len(X_test) - 1)
test_img = X_test[idx]

# Expand dimensions to match model input shape and predict the label
prediction = model.predict(test_img[None, ...], verbose=0)
pred_label = labels[np.argmax(prediction)]

# Display the selected image with the predicted label
plt.imshow(test_img.squeeze(), cmap='gray' if test_img.shape[-1] == 1 else None)
plt.title(f"Predicted: {pred_label}")
plt.axis('off')
plt.show()
