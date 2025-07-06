import random
import numpy as np
import matplotlib.pyplot as plt

# Select random test image
i = random.randrange(len(X_test))
x = X_test[i]
y = model.predict(x[np.newaxis, ...], verbose=0)[0]

# Get predicted label and confidence
j = np.argmax(y)
lbl = labels[j]
conf = y[j] * 100

# Display the image with prediction
plt.imshow(x.squeeze(), cmap='gray' if x.shape[-1] == 1 else None)
plt.title(f"{lbl} ({conf:.2f}%)")
plt.axis('off')
plt.tight_layout()
plt.show()
