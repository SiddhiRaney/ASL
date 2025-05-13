import random
import numpy as np
import matplotlib.pyplot as plt

# Randomly select one test image
test_image = X_test[random.randint(0, len(X_test) - 1)]

# Predict the label
pred = model.predict(test_image[None, ...], verbose=0)
predicted_label = labels[np.argmax(pred)]

# Display the image and predicted label
plt.imshow(test_image)
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
