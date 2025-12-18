import numpy as np

#compute the "roughness" of images by looking at smaller blocks
def block_roughness_features(image, block_size):
    A = np.asarray(image, dtype=float)
    #eventhough H and W are 64, not a good cofing practice to assume that. so we get the shape dynamically.
    H, W = A.shape
    rough = []
    #loop over blocks row-wise and column-wise
    for i in range(1, H, block_size):
        for j in range(1, W, block_size):
            #slices out one block of size block_size x block_size
            B = A[i:i+block_size, j:j+block_size]
            #compute the horizontal finite difference wirhin the block
            dx = B[:, 1:] - B[:, :-1]
            #compute the vertical finite difference within the block
            dy = B[1:, :] - B[:-1, :]
            #now we have how fast intensity is changing in x and y direction within the block.
            #we now compute the mean squared finite difference as a measure of roughness within the block
            #mean squared finite difference between adjacent pixels in x and y direction and sum them up to get roughness
            r = np.mean(dx*dx) + np.mean(dy*dy)
            rough.append(r)
    rough = np.array(rough, dtype=float)
    return np.array([rough.mean(), rough.std(), rough.max()], dtype=float)

def svd_features(image,p,tol=1e-12):
    """
    Compute the top p singular values of the input image.

    Parameters:
    image (np.ndarray): 2D array representing the grayscale image.
    p (int): Number of top singular values to return.

    Returns:
    np.ndarray: 1D array of the top p singular values.
    """
    #singular value decomposition and only get the singular values. U and V transpose are not needed.
    S = np.linalg.svd(image, full_matrices=True, compute_uv=False)

    energy = S**2
    total_energy = np.sum(energy)
    #cumulative sum of squared singular values. this is an array where each element at index i represents the sum of squared singular values from index 0 to i.
    cumulative_energy = np.cumsum(energy) 
    normalized_cumulative_energy = cumulative_energy / total_energy if total_energy >= tol else cumulative_energy  #Good old normalization. Now all the arrays values are between 0 and 1 and add up to 1.

    #get the top p singular values
    top_p_singular_values = S[:p]/np.sum(S) if np.sum(S) >= tol else S[:p]  # normalized singular values
    r95 = float(np.searchsorted(normalized_cumulative_energy, 0.95) + 1)
    r99 = float(np.searchsorted(normalized_cumulative_energy, 0.99) + 1)

    # get roughness features, trial and error shows block size of 8 works well.
    rough_feat = block_roughness_features(image, block_size=8)
    return np.concatenate([top_p_singular_values, np.array([r95, r99], dtype=np.float32), rough_feat])


def lda_train(X,y):
    """
    Train a Linear Discriminant Analysis (LDA) model.

    Parameters:
    X (np.ndarray): 2D array of shape (N, D) where N is the number of samples and D is the feature dimension.
    y (np.ndarray): 1D array of shape (N,) containing class labels (0 or 1).

    Returns:
    tuple: A tuple containing the projection vector w and the bias term b.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Split into classes based on human vs AI labels. X0 is the feature vectors for class 0, X1 for class 1.
    X0 = X[y == 0]
    X1 = X[y == 1]

    # Class means
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    #Now we create the within class scatter matrix Sw
    X0_centered = X0 - mu0
    X1_centered = X1 - mu1
    Sw = (X0_centered.T @ X0_centered) + (X1_centered.T @ X1_centered) #the vectorzed outer proiduct computation to get Sw

    #Now the between class scatter matrix Sb
    diff = mu1 - mu0                 
    Sb = diff[:, None] @ diff[None, :] #Here we multply the columnm vector (d,1) with the row vector (1,d) to get a (d,d) matrix.
    
    #We seek a projection vector w that maximizes the Rayleigh quotient which is the ratio of between-class variance to within-class variance.
    #Since this is a tw0-class problem we do not need to solve a generalized eigenvalue problem. We can directly compute w as follows:
    # w is proportional to Sw^-1 * (mu1 - mu0). Taking inverse is expensive and numerically unstable, so we solve the linear system Sw * w = (mu1 - mu0)
    #But before that we need to add a small regularization term to avoid numerical instability in case Sw is singular or ill-conditioned.
    d = Sw.shape[0]
    lam = 1e-6 * np.trace(Sw) / d if np.trace(Sw) > 0 else 1e-6
    Sw = Sw + lam * np.eye(d)

    #Now we finally solve for w
    w = np.linalg.solve(Sw, mu1 - mu0)

    #Now we implement the classification rule in 1d.
    #To do so we project a feature vector x to a scalar z = w^T * x. So we compute mo and m1 which are the projected means of each class.
    m0 = w.T @ mu0
    m1 = w.T @ mu1

    #Now the natural decision boundary is at the midpoint between m0 and m1.
    tau = (m0 + m1) / 2

    #LDA training is done. Voila!!
    return w , tau

def lda_predict(X, w, threshold):
    return (X @ w >= threshold).astype(int)

def _example_run():
    """Run a tiny end-to-end test on the example dataset, if available.

    This function is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data.npz")
    except OSError:
        print("No example data file 'project_data.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Sanity check shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    p = min(32, min(X_train.shape[1], X_train.shape[2]))
    print(f"Using p = {p} leading singular values for features.")

    # Build feature matrices
    def build_features(X):
        feats = []
        for img in X:
            feats.append(svd_features(img, p))
        return np.vstack(feats)

    try:
        Xf_train = build_features(X_train)
        Xf_test = build_features(X_test)
    except NotImplementedError:
        print("Implement 'svd_features' first to run this example.")
        return

    print("Feature dimension:", Xf_train.shape[1])

    try:
        w, threshold = lda_train(Xf_train, y_train)
    except NotImplementedError:
        print("Implement 'lda_train' first to run this example.")
        return

    try:
        y_pred = lda_predict(Xf_test, w, threshold)
    except NotImplementedError:
        print("Implement 'lda_predict' first to run this example.")
        return

    accuracy = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    # This allows students to run a quick local smoke test.
    _example_run()