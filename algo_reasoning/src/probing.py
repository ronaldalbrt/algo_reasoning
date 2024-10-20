import torch

def probe_array(A_pos):
    """Constructs an `array` probe."""
    probe = torch.arange(A_pos.size(0))
    for i in range(1, A_pos.size(0)):
        probe[A_pos[i]] = A_pos[i - 1]

    return probe

def mask_one(i, n):
    """Constructs a `mask_one` probe."""
    assert n > i
    probe = torch.zeros(n)
    probe[i] = 1
    return probe


def heap(A_pos, heap_size):
    """Constructs a `heap` probe."""
    assert heap_size > 0
    
    probe = torch.arange(A_pos.size(0))
    
    for i in range(1, heap_size):
        probe[A_pos[i]] = A_pos[(i - 1) // 2]
    
    return probe
