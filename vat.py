import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE hyper-parameters we use in VAT
# n_power: a number of power iteration for approximation of r_vadv
# XI: a small float for the approx. of the finite difference method
# epsilon: the value for how much deviate from original data point X


class VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, model, n_power, XI, epsilon):
        super(VAT, self).__init__()
        self.model = model
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon

    def forward(self, X, logit, seq_lengths):
        vat_loss = virtual_adversarial_loss(X, logit, self.model, self.n_power,
                                            self.XI, self.epsilon, seq_lengths)
        return vat_loss  # already averaged


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
            d.size(0), 1, 1)

    # print(d_abs_max.size())
    d /= (1e-12 + d_abs_max)
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
    # print(torch.norm(d.view(d.size(0), -1), dim=1))
    return d


def generate_virtual_adversarial_perturbation(x, logit, model, n_power, XI,
                                              epsilon, seq_lengths):
    d = torch.randn_like(model.get_emb(x))

    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        num_sample = x.shape[1]
        hidden = model.init_hidden(num_sample)
        logit_m, _ = model(x, hidden, seq_lengths, vat=True, d=d)
        dist = kl_divergence_with_logit(logit, logit_m)
        grad = torch.autograd.grad(dist, [d])[0]
        d = grad.detach()

    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, model, n_power, XI, epsilon, seq_lengths):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, model,
                                                       n_power, XI, epsilon, seq_lengths)
    logit_p = logit.detach()
    num_sample = x.shape[1]
    hidden = model.init_hidden(num_sample)
    logit_m, _ = model(x, hidden, seq_lengths, vat=True, d=r_vadv)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return loss