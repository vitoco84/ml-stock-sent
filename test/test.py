from src.utils import is_cuda_available


def test_is_cuda_available():
    assert is_cuda_available() == True
