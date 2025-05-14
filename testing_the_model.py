import random
import numpy as np
import matplotlib.pyplot as plt

# Randomly select one test image
test_image = random.choice(X_test)

# Predict the label
pred = model.predict(np.expand_dims(test_image, axis=0))
predicted_label = labels[np.argmax(pred)]

# Display the image and predicted label
plt.imshow(test_image.squeeze())  # Use .squeeze() if image has a singleton channel dimension
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
