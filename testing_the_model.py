import random

# Pick a random test image
idx = random.randint(0, len(X_test))
img = X_test[idx]

# Predict
pred = model.predict(np.expand_dims(img, axis=0))
pred_label = labels[np.argmax(pred)]

# Show image and prediction
import matplotlib.pyplot as plt
plt.imshow(img)
plt.title(f"Predicted: {pred_label}")
plt.show()
