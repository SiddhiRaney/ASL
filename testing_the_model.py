import random
import numpy as np
import matplotlib.pyplot as plt

# Randomly select a test image
random_index = random.randint(0, len(X_test) - 1)
test_image = X_test[random_index]

# Add batch dimension and predict
prediction = model.predict(np.expand_dims(test_image, axis=0), verbose=0)[0]

# Extract predicted label and confidence
predicted_index = np.argmax(prediction)
predicted_label = labels[predicted_index]
confidence = prediction[predicted_index] * 100

# Prepare image for display
display_image = np.squeeze(test_image)
is_grayscale = display_image.ndim == 2 or (display_image.ndim == 3 and display_image.shape[-1] == 1)

# Show the image with predicted label and confidence
plt.imshow(display_image, cmap='gray' if is_grayscale else None)
plt.title(f"Predicted: {predicted_label} ({confidence:.2f}%)")
plt.axis('off')
plt.tight_layout()
plt.show()
