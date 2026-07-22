"""Tests for project_2d: a from-scratch 2D PCA over plain Python/numpy, no scikit-learn."""
import math
import pytest
from src.rag.visualize import project_2d

def test_returns_one_point_per_vector():
    vectors = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    points = project_2d(vectors)
    assert len(points) == 4
    assert all(len(p) == 2 for p in points)

def test_identical_vectors_collapse_to_origin():
    # No variance at all -> every point projects to (0, 0) (mean-centered, zero spread).
    vectors = [[1.0, 2.0, 3.0]] * 5
    points = project_2d(vectors)
    for x, y in points:
        assert math.isclose(x, 0.0, abs_tol=1e-9)
        assert math.isclose(y, 0.0, abs_tol=1e-9)

def test_separates_two_distinct_clusters():
    # Two tight clusters far apart in a high-dim space must land far apart in 2D too.
    cluster_a = [[0.0, 0.0, 0.0, 0.0]] * 3
    cluster_b = [[10.0, 10.0, 10.0, 10.0]] * 3
    points = project_2d(cluster_a + cluster_b)
    a_pts, b_pts = points[:3], points[3:]
    def dist(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])
    # Any point in cluster A must be much closer to another A point than to any B point.
    assert dist(a_pts[0], a_pts[1]) < dist(a_pts[0], b_pts[0])

def test_raises_on_fewer_than_two_vectors():
    with pytest.raises(ValueError):
        project_2d([[1.0, 2.0]])

def test_raises_on_empty_input():
    with pytest.raises(ValueError):
        project_2d([])
