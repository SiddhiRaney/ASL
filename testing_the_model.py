import random
import numpy as np
import matplotlib.pyplot as plt

# Pick random test image and predict
img = X_test[random.randint(0, len(X_test) - 1)]
pred = model.predict(img[None], verbose=0)[0]
label, conf = labels[np.argmax(pred)], np.max(pred) * 100

# Show image with prediction
plt.imshow(img.squeeze(), cmap='gray' if img.shape[-1] == 1 else None)
plt.title(f"{label} ({conf:.2f}%)")
plt.axis('off')
plt.tight_layout()
plt.show()
