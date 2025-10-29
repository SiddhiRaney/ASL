import random
import numpy as np
import matplotlib.pyplot as plt

# Randomly pick one test sample
idx = random.randrange(len(X_test))
image = X_test[idx]

# Predict once and get top result
probs = model.predict(image[np.newaxis], verbose=0)[0]
pred_idx = np.argmax(probs)
label = labels[pred_idx]
confidence = probs[pred_idx] * 100

# Display the result
plt.imshow(image.squeeze(), cmap='gray' if image.ndim == 3 and image.shape[-1] == 1 else None)
plt.title(f"{label} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
