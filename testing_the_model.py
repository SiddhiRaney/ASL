import random
import numpy as np
import matplotlib.pyplot as plt

# Pick a random test image and predict its label
idx = random.randrange(len(X_test))
img = X_test[idx]
pred = model.predict(img[np.newaxis, ...], verbose=0)[0]

# Get predicted label and confidence
pred_idx = np.argmax(pred)
label = labels[pred_idx]
confidence = pred[pred_idx] * 100

# Display the image with prediction
plt.imshow(img.squeeze(), cmap='gray' if img.shape[-1] == 1 else None)
plt.title(f"{label} ({confidence:.2f}%)")
plt.axis('off')
plt.tight_layout()
plt.show()
