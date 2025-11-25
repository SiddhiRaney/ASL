import numpy as np
import matplotlib.pyplot as plt
import random

def show_random_pred(model, X, lbls):
    # pick image
    i = random.randint(0, len(X)-1)
    im = X[i]

    # prediction
    pr = model.predict(im[None], verbose=0)[0]
    c = pr.argmax()
    conf = pr[c] * 100

    # plot
    plt.figure(figsize=(4,4))
    plt.imshow(im.squeeze(), cmap='gray' if im.ndim==3 and im.shape[-1]==1 else None)
    plt.title(f"{lbls[c]} — {conf:.2f}%", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call twice (or however many times you want)
show_random_pred(model, X_test, labels)
show_random_pred(model, X_test, labels)
