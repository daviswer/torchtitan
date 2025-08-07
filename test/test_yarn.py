import pytest
import torch

from torchtitan.models.llama.model import (
    _precompute_freqs_cis_original_llama,
    precompute_freqs_cis,
)


def test_no_yarn_limit_equivalence():
    """
    Expect the same outputs when original_seq_len == seq_len
    """
    dim = 64
    seq_len = original_seq_len = 2048
    freqs_llama = _precompute_freqs_cis_original_llama(dim=dim, end=seq_len)
    freqs_yarn = precompute_freqs_cis(
        dim=dim, seq_len=seq_len, original_seq_len=original_seq_len
    )
    torch.testing.assert_close(freqs_llama, freqs_yarn)


def test_yarn_different():
    """
    Expect different outputs when original_seq_len != seq_len
    """
    dim = 64
    seq_len = 2048
    original_seq_len = 1024
    freqs_llama = _precompute_freqs_cis_original_llama(dim=dim, end=seq_len)
    freqs_yarn = precompute_freqs_cis(
        dim=dim, seq_len=seq_len, original_seq_len=original_seq_len
    )
    with pytest.raises(AssertionError, match="Mismatched elements"):
        torch.testing.assert_close(freqs_llama, freqs_yarn)
