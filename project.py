import numpy as np



def svd_features(image, p):
    s = np.linalg.svd(image, full_matrices=False, compute_uv=False)
    s = s[:p]
    s = s / np.sum(s) if np.sum(s) > 0 else s
    return np.concatenate([s, np.array([p, p])])

def lda_train(X, y):
    X0, X1 = X[y==0], X[y==1]
    mu0, mu1 = X0.mean(0), X1.mean(0)
    S0 = (X0 - mu0).T @ (X0 - mu0)
    S1 = (X1 - mu1).T @ (X1 - mu1)
    Sw = S0 + S1 + 1e-6*np.eye(X.shape[1])
    w = np.linalg.solve(Sw, mu1 - mu0)
    thresh = 0.5*(mu0@w + mu1@w)
    return w, thresh


def lda_predict(X, w, threshold):
    return (X @ w >= threshold).astype(int)
def _example_run():
    """Run a tiny end-to-end test on the example dataset, if available.

    This function is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example data file 'project_data_example.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Sanity check shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    p = min(64, min(X_train.shape[1], X_train.shape[2]))
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