import numpy as np
import matplotlib.pyplot as plt
import random

# Pick a random test image
idx = random.randint(0, len(X_test) - 1)
img = X_test[idx]

# Predict probabilities for the chosen image
probs = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
pred_idx = np.argmax(probs)
label = labels[pred_idx]
confidence = probs[pred_idx] * 100

# Visualization setup
plt.figure(figsize=(4, 4))
plt.imshow(img.squeeze(), cmap='gray' if img.ndim == 3 and img.shape[-1] == 1 else None)
plt.title(f"{label} — {confidence:.2f}%", fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.show()

idx = np.random.randint(len(X_test))
img = X_test[idx]

probs = model.predict(img[None], verbose=0)[0]
p = probs.argmax()

plt.figure(figsize=(4,4))
plt.imshow(img.squeeze(), cmap='gray' if img.ndim==3 and img.shape[-1]==1 else None)
plt.title(f"{labels[p]} — {probs[p]*100:.2f}%", fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.show()
