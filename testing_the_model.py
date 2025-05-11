import random
import numpy as np
import matplotlib.pyplot as plt

# Select a random image from the test set
random_index = random.randint(0, len(X_test) - 1)
test_image = X_test[random_index]

# Make prediction
prediction = model.predict(np.expand_dims(test_image, axis=0), verbose=0)
predicted_label = labels[np.argmax(prediction)]

# Display image with prediction
plt.imshow(test_image)
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')  # Hide axes for cleaner display
plt.show()
