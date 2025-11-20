# tests/array_backend/test_array_backend_utils.py
import numpy as np
import pytest

from probpipe.array_backend import utils as U


def test_ensure_real_scalar_from_python_scalar():
    assert U._ensure_real_scalar(3) == 3
    assert U._ensure_real_scalar(3.5) == 3.5
    assert U._ensure_real_scalar(3, as_array=True) == np.array(3)
    assert U._ensure_real_scalar(3.5, as_array=True) == np.array(3.5)


def test_ensure_real_scalar_from_numpy_scalar_and_0d():
    a = np.float32(2.0)
    assert isinstance(U._ensure_real_scalar(a), float)
    assert U._ensure_real_scalar(a, as_array=True) == np.asarray([a])

    b = np.array(4.0)
    assert U._ensure_real_scalar(b) == 4.0
    assert U._ensure_real_scalar(b, as_array=True) == b


@pytest.mark.parametrize(
    "non_scalar_input",
    [[1,2], np.arange(2), np.identity(2)]
)
def test_ensure_real_scalar_rejects_multiple_elements(non_scalar_input):
    with pytest.raises(ValueError):
        U._ensure_real_scalar(non_scalar_input)


def test_ensure_real_scalar_rejects_complex():
    with pytest.raises(ValueError):
        U._ensure_real_scalar(1 + 2j)
    with pytest.raises(ValueError):
        U._ensure_real_scalar(np.array(1 + 0j))


def test_ensure_batch_real_scalar_single_and_array():
    a = U._ensure_batch_real_scalar(2)
    assert a.shape == (1,)
    assert a[0] == 2

    b = U._ensure_batch_real_scalar(np.array([1.0, 2.0]))
    assert b.shape == (2,)
    assert np.allclose(b, np.array([1.0, 2.0]))


def test_ensure_vector_scalar_and_1d_and_2d():
    v0 = U._ensure_vector(5)
    assert v0.shape == (1,)
    v1 = U._ensure_vector([1, 2, 3])
    assert v1.shape == (3,)
    vcol = U._ensure_vector([1, 2], as_column=True)
    assert vcol.shape == (2, 1)
    # 2D (1,n)
    v2 = U._ensure_vector(np.array([[1, 2, 3]]), length=3)
    assert v2.shape == (3,)


def test_ensure_vector_length_check():
    with pytest.raises(ValueError):
        U._ensure_vector([1, 2, 3], length=2)


def test_ensure_matrix_scalar_1d_2d():
    m0 = U._ensure_matrix(2)
    assert m0.shape == (1, 1)
    m1 = U._ensure_matrix([1, 2], as_row_matrix=True)
    assert m1.shape == (1, 2)
    m2 = U._ensure_matrix([1, 2], as_row_matrix=False)
    assert m2.shape == (2, 1)
    m3 = U._ensure_matrix(np.array([[1, 2], [3, 4]]))
    assert m3.shape == (2, 2)


def test_ensure_matrix_rejects_ndim_gt3():
    with pytest.raises(ValueError):
        U._ensure_matrix(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        U._ensure_matrix(np.ones((1, 1, 1, 1)))


def test_ensure_matrix_dim_checks():
    with pytest.raises(ValueError):
        m0 = U._ensure_matrix(2, num_rows=1, num_cols=2)
    with pytest.raises(ValueError):
        m1 = U._ensure_matrix([1, 2], as_row_matrix=True, num_rows=2)
    with pytest.raises(ValueError):
        m1 = U._ensure_matrix([1, 2], as_row_matrix=False, num_cols=2)


def test_ensure_square_matrix():
    M = np.eye(3)
    out = U._ensure_square_matrix(M)
    assert out.shape == (3, 3)
    with pytest.raises(ValueError):
        U._ensure_square_matrix(np.array([[1, 2], [3, 4]]), n=3)
    with pytest.raises(ValueError):
        U._ensure_square_matrix(np.array([1, 2, 3]))


def test_ensure_batch_array_basic_and_value_shape():
    a = np.array([1,2,3])
    b = a.copy().reshape((1,3))
    c = U._ensure_batch_array(a, copy=True)
    assert np.allclose(b, c)

    assert U._ensure_batch_array(5) == np.array([[5]])

    mat = np.zeros((2, 2))
    batch_mat = U._ensure_batch_array(mat, value_shape=(2, 2))
    assert batch_mat.shape == (1, 2, 2)

    arr = np.ones((1,3,4))
    assert U._ensure_batch_array(arr, copy=True, value_shape=(3,4)).shape == (1,3,4)

    with pytest.raises(ValueError):
        U._ensure_batch_array(arr, copy=True, value_shape=(4,3))


def test_ensure_batch_real_scalar():
    with pytest.raises(ValueError):
        U._ensure_batch_real_scalar(np.zeros((2, 2)))

def test_ensure_batch_vector():
    v = U._ensure_batch_vector([1, 2, 3])
    assert v.shape == (1, 3)
    batch_v = U._ensure_batch_vector(np.array([[1, 2], [3, 4]]), length=2)
    assert batch_v.shape == (2, 2)
    assert U._ensure_batch_vector(1, length=1) == np.array([[1]])

    with pytest.raises(ValueError):
       U._ensure_batch_vector(np.zeros((2,2,2))) 
    with pytest.raises(ValueError):
       U._ensure_batch_vector(np.zeros((3,2)), length=3) 


def test_ensure_batch_matrix():
    assert U._ensure_batch_matrix(1) == np.array(1).reshape((1,1,1))
    m1 = U._ensure_batch_matrix([1, 2, 3], as_row_matrix=True)
    assert m1.shape == (1, 1, 3)
    m2 = U._ensure_batch_matrix([1, 2, 3], as_row_matrix=False)
    assert m2.shape == (1, 3, 1)

    m3 = np.zeros((2, 3))
    bm = U._ensure_batch_matrix(m3)
    assert bm.shape == (1, 2, 3)
    batch_m = U._ensure_batch_matrix(np.zeros((4, 2, 3)), num_rows=2, num_cols=3)
    assert batch_m.shape == (4, 2, 3)

    with pytest.raises(ValueError):
       U._ensure_batch_matrix(np.zeros((4, 2, 3)), num_rows=3)
    with pytest.raises(ValueError):
       U._ensure_batch_matrix(np.zeros((4, 2, 3, 1))) 


def test_copy_semantics_ensure_vector_matrix_batch():
    arr = np.array([1.0, 2.0, 3.0])
    v_copy = U._ensure_vector(arr, copy=True)
    assert not v_copy is arr
    v_view = U._ensure_vector(arr, copy=False)
    # may be same object or view; at least ensure returned values equal
    assert np.allclose(np.ravel(v_view), arr)

    mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    m_copy = U._ensure_matrix(mat, copy=True)
    assert not m_copy is mat
    m_nocopy = U._ensure_matrix(mat, copy=False)
    assert np.allclose(m_nocopy, mat)

    batch = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_copy = U._ensure_batch_array(batch, copy=True)
    assert not b_copy is batch
    b_nocopy = U._ensure_batch_array(batch, copy=False)
    assert np.allclose(b_nocopy, batch)