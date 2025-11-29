import numpy as np
import matplotlib.pyplot as plt
import random

def show_rand_pred(model, data, names):
    # choose random idx
    j = random.randrange(len(data))
    img = data[j]

    # predict
    p = model.predict(img[None], verbose=0)[0]
    ci = p.argmax()
    acc = p[ci] * 100

    # plot
    plt.figure(figsize=(4,4))
    plt.imshow(img.squeeze(), cmap="gray" if img.ndim == 3 and img.shape[-1] == 1 else None)
    plt.title(f"{names[ci]} — {acc:.2f}%", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# calls
show_rand_pred(model, X_test, labels)
show_rand_pred(model, X_test, labels)
