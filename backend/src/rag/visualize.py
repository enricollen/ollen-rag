"""2D PCA projection of embedding vectors for the Indices Visualizer — plain numpy, no
scikit-learn/umap-learn dependency. Eigendecomposition of the covariance matrix (numpy's
eigh, since the covariance matrix is always symmetric) is cheaper and more stable here
than a full SVD of the data matrix."""
import numpy as np

def project_2d(vectors: list[list[float]]) -> list[tuple[float, float]]:
    """Project n vectors (each dimension d) onto their top-2 principal components.

    Mean-centers the input, eigendecomposes the (d, d) covariance matrix, and projects
    onto the eigenvectors of the two largest eigenvalues. Raises ValueError for fewer
    than 2 input vectors (no meaningful spread to project).
    """
    if len(vectors) < 2:
        raise ValueError("project_2d requires at least 2 vectors")
    matrix = np.asarray(vectors, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    # eigh returns eigenvalues ascending; the last two columns are the top-2 components.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    top2 = eigenvectors[:, -2:]
    projected = centered @ top2
    return [(float(x), float(y)) for x, y in projected]
