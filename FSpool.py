from imports import *


class FSPool(nn.Module):
    """
        Featurewise sort pooling. From:
        FSPool: Learning Set Representations with Featurewise Sort Pooling.
        Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
        https://arxiv.org/abs/1906.02795
        https://github.com/Cyanogenoid/fspool
    """

    def __init__(self, in_channels, n_pieces, relaxed=False):
        """
        in_channels: Number of channels in input
        n_pieces: Number of pieces in piecewise linear
        relaxed: Use sorting networks relaxation instead of traditional sorting
        """
        super().__init__()
        self.n_pieces = n_pieces
        self.weight = nn.Parameter(torch.zeros(in_channels, n_pieces + 1))
        self.relaxed = relaxed

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x, n=None):
        """ FSPool
        x: FloatTensor of shape (batch_size, in_channels, set size).
        This should contain the features of the elements in the set.
        Variable set sizes should be padded to the maximum set size in the batch with 0s.
        n: LongTensor of shape (batch_size).
        This tensor contains the sizes of each set in the batch.
        If not specified, assumes that every set has the same size of x.size(2).
        Note that n.max() should never be greater than x.size(2), i.e. the specified set size in the
        n tensor must not be greater than the number of elements stored in the x tensor.
        Returns: pooled input x, used permutation matrix perm
        """
        assert x.size(1) == self.weight.size(
            0
        ), "incorrect number of input channels in weight"
        # can call withtout length tensor, uses same length for all sets in the batch
        if n is None:
            n = x.new(x.size(0)).fill_(x.size(2)).long()
        # create tensor of ratios $r$
        sizes, mask = fill_sizes(n, x)
        mask = mask.expand_as(x)

        # turn continuous into concrete weights
        weight = self.determine_weight(sizes)

        # make sure that fill value isn't affecting sort result
        # sort is descending, so put unreasonably low value in places to be masked away
        x = x + (1 - mask).float() * -99999
        if self.relaxed:
            x, perm = cont_sort(x, temp=self.relaxed)
        else:
            x, perm = x.sort(dim=2, descending=True)

        x = (x * weight * mask.float()).sum(dim=2)
        return x, perm

    def forward_transpose(self, x, perm, n=None):
        """ FSUnpool 
        x: FloatTensor of shape (batch_size, in_channels)
        perm: Permutation matrix returned by forward function.
        n: LongTensor fo shape (batch_size)
        """
        if n is None:
            n = x.new(x.size(0)).fill_(perm.size(2)).long()
        sizes, mask = fill_sizes(n)
        mask = mask.expand(mask.size(0), x.size(1), mask.size(2))

        weight = self.determine_weight(sizes)

        x = x.unsqueeze(2) * weight * mask.float()

        if self.relaxed:
            x, _ = cont_sort(x, perm)
        else:
            x = x.scatter(2, perm, x)
        return x, mask

    def determine_weight(self, sizes):
        """
            Piecewise linear function. Evaluates f at the ratios in sizes.
            This should be a faster implementation than doing the sum over max terms, since we know that most terms in it are 0.
        """
        # share same sequence length within each sample, so copy weighht across batch dim
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(sizes.size(0), weight.size(1), weight.size(2))

        # linspace [0, 1] -> linspace [0, n_pieces]
        index = self.n_pieces * sizes
        index = index.unsqueeze(1)
        index = index.expand(index.size(0), weight.size(1), index.size(2))

        # points in the weight vector to the left and right
        idx = index.long()
        frac = index.frac()
        left = weight.gather(2, idx)
        right = weight.gather(2, (idx + 1).clamp(max=self.n_pieces))

        # interpolate between left and right point
        return (1 - frac) * left + frac * right


def fill_sizes(sizes, x=None):
    """
        sizes is a LongTensor of size [batch_size], containing the set sizes.
        Each set size n is turned into [0/(n-1), 1/(n-1), ..., (n-2)/(n-1), 1, 0, 0, ..., 0, 0].
        These are the ratios r at which f is evaluated at.
        The 0s at the end are there for padding to the largest n in the batch.
        If the input set x is passed in, it guarantees that the mask is the correct size even when sizes.max()
        is less than x.size(), which can be a case if there is at least one padding element in each set in the batch.
    """
    if x is not None:
        max_size = x.size(2)
    else:
        max_size = sizes.max()
    size_tensor = sizes.new(sizes.size(0), max_size).float().fill_(-1)

    size_tensor = torch.arange(end=max_size, device=sizes.device, dtype=torch.float32)
    size_tensor = size_tensor.unsqueeze(0) / (sizes.float() - 1).clamp(min=1).unsqueeze(
        1
    )

    mask = size_tensor <= 1
    mask = mask.unsqueeze(1)

    return size_tensor.clamp(max=1), mask.float()


def deterministic_sort(s, tau):
    """
    "Stochastic Optimization of Sorting Networks via Continuous Relaxations" https://openreview.net/forum?id=H1eSS3CcKX
    Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon
    s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar.
    """
    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=s.device)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, one.transpose(0, 1)))
    scaling = (n + 1 - 2 * (torch.arange(n, device=s.device) + 1)).type(torch.float32)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def cont_sort(x, perm=None, temp=1):
    """ Helper function that calls deterministic_sort with the right shape.
    Since it assumes a shape of (batch_size, n, 1) while the input x is of shape (batch_size, channels, n),
    we can get this to the right shape by merging the first two dimensions.
    If an existing perm is passed in, we compute the "inverse" (transpose of perm) and just use that to unsort x.
    """
    original_size = x.size()
    x = x.view(-1, x.size(2), 1)
    if perm is None:
        perm = deterministic_sort(x, temp)
    else:
        perm = perm.transpose(1, 2)
    x = perm.matmul(x)
    x = x.view(original_size)
    return x, perm







class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, dim_input, dim_output, num_outputs=1,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output)
                )
        # self.dec = nn.Sequential(
        #         PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         nn.Linear(dim_hidden, dim_output)
        #         )

    def forward(self, X):
        #return self.dec(self.enc(X))
        return self.enc(X)