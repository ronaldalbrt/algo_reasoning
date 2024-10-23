import torch
from algo_reasoning.src.specs import OutputClass

def probe_array(A_pos):
    """Constructs an `array` probe."""
    probe = torch.arange(A_pos.size(0))
    for i in range(1, A_pos.size(0)):
        probe[A_pos[i]] = A_pos[i - 1]

    return probe

def array_cat(A, n):
    """Constructs an `array_cat` probe."""
    assert n > 0
    probe = torch.zeros((A.size(0), n))

    for i in range(A.size(0)):
        probe[i, A[i].item()] = 1

    return probe


def mask_one(i, n):
    """Constructs a `mask_one` probe."""
    assert n > i
    probe = torch.zeros(n)
    probe[i] = 1
    return probe

def graph(A):
    """Constructs a `graph` probe."""
    probe = ((A + torch.eye(A.size(0))) != 0) * 1.0

    return probe


def heap(A_pos, heap_size):
    """Constructs a `heap` probe."""
    assert heap_size > 0
    
    probe = torch.arange(A_pos.size(0))
    
    for i in range(1, heap_size):
        probe[A_pos[i]] = A_pos[(i - 1) // 2]
    
    return probe

def strings_id(T_pos, P_pos):
    """Constructs a `strings_id` probe."""
    probe_T = torch.zeros(T_pos.size(0))
    probe_P = torch.ones(P_pos.size(0))

    return torch.concatenate([probe_T, probe_P])


def strings_pair(pair_probe):
    """Constructs a `strings_pair` probe."""
    n = pair_probe.size(0)
    m = pair_probe.size(1)
    probe_ret = torch.zeros((n + m, n + m))

    for i in range(0, n):
        for j in range(0, m):
            probe_ret[i, j + n] = pair_probe[i, j].item()
    return probe_ret


def strings_pair_cat(pair_probe, nb_classes):
    """Constructs a `strings_pair_cat` probe."""
    assert nb_classes > 0
    n = pair_probe.size(0)
    m = pair_probe.size(1)

    # Add an extra class for 'this cell left blank.'
    probe_ret = torch.zeros((n + m, n + m, nb_classes + 1))
    for i in range(0, n):
        for j in range(0, m):
            probe_ret[i, j + n, int(pair_probe[i, j])] = OutputClass.POSITIVE

    # Fill the blank cells.
    for i_1 in range(0, n):
        for i_2 in range(0, n):
            probe_ret[i_1, i_2, nb_classes] = OutputClass.MASKED
    for j_1 in range(0, m):
        for x in range(0, n + m):
            probe_ret[j_1 + n, x, nb_classes] = OutputClass.MASKED
    return probe_ret

def strings_pi(T_pos, P_pos, pi):
    """Constructs a `strings_pi` probe."""
    probe = torch.arange(T_pos.size(0) + P_pos.size(0))
    
    for j in range(P_pos.size(0)):
        probe[T_pos.size(0) + P_pos[j]] = T_pos.size(0) + pi[P_pos[j]]
    
    return probe

def strings_pred(T_pos, P_pos):
    """Constructs a `strings_pred` probe."""
    probe = torch.arange(T_pos.size(0) + P_pos.size(0))

    for i in range(1, T_pos.size(0)):
        probe[T_pos[i]] = T_pos[i - 1].item()
    for j in range(1, P_pos.size(0)):
        probe[T_pos.size(0) + P_pos[j]] = T_pos.size(0) + P_pos[j - 1].item()
    return probe