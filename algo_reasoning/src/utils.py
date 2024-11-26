import torch
import math

def logsumexp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)
    
def normal_log_pdf(sample, mu, sigma2):
    normalization_term = torch.log(2*torch.pi*sigma2)

    return -0.5*(((sample - mu)**2)/(sigma2) + normalization_term)

def multivariatenormal_log_pdf(sample, mu, cov):
    n_dim = sample.size(1)

    det_log = torch.logdet(cov) 
    det_log = det_log if not (torch.isnan(det_log).item() or det_log.item() == float('-inf')) else 0

    normalization_term = n_dim*math.log(2*torch.pi) + det_log
    inv_cov = torch.linalg.pinv(cov)

    return -0.5*((((sample - mu)@inv_cov) * (sample - mu)).sum(dim=1) + normalization_term)