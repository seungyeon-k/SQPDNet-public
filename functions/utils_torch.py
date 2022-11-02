import torch

def quats_to_matrices_torch(quaternions):
    # An unit quaternion is q = xi + yj + zk + w
    x = quaternions[:, 0]
    y = quaternions[:, 1]
    z = quaternions[:, 2]
    w = quaternions[:, 3]

    # Initialize
    K = quaternions.shape[0]
    R = quaternions.new_zeros((K, 3, 3))

    xx = x**2
    yy = y**2
    zz = z**2
    ww = w**2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * x * y
    xz = s[:, 0] * x * z
    xw = s[:, 0] * x * w
    yz = s[:, 0] * y * z
    yw = s[:, 0] * y * w
    zw = s[:, 0] * z * w

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R

def thetas_to_matrices_torch(theta):
    # initialize
    K = theta.shape[0]
    R = theta.new_zeros((K, 3, 3))

    cos = theta[:,0]
    sin = theta[:,1]

    idxs = torch.arange(K).to(theta.device)
    R[idxs, 0, 0] = cos
    R[idxs, 0, 1] = -sin
    R[idxs, 1, 0] = sin
    R[idxs, 1, 1] = cos
    R[idxs, 2, 2] = 1

    return R

def matrices_to_quats_torch(R):
    original_ndim = R.ndim
    if original_ndim == 2:
        R = R.unsqueeze(0).to(R)
    elif original_ndim == 3:
        pass
    else:
        raise NotImplementedError("Dimension of matrices must be 2 or 3")

    qr = 0.5 * torch.sqrt(1+torch.einsum('ijj->i', R)).unsqueeze(1)
    qi = 1/(4*qr) * (R[:, 2,1] - R[:, 1,2]).unsqueeze(1)
    qj = 1/(4*qr) * (R[:, 0,2] - R[:, 2,0]).unsqueeze(1)
    qk = 1/(4*qr) * (R[:, 1,0] - R[:, 0,1]).unsqueeze(1)

    if original_ndim == 2:
        R = R.squeeze(0)

    return torch.cat([qi, qj, qk, qr], dim=1).to(R)

def get_device_info(x):
    cuda_check = x.is_cuda
    if cuda_check:
        device = "cuda:{}".format(x.get_device())
    else:
        device = 'cpu'
    return device

def skew(w):
    n = w.shape[0]
    device = get_device_info(w)
    if w.shape == (n, 3, 3):
        W = torch.cat([-w[:, 1, 2].unsqueeze(-1),
                       w[:, 0, 2].unsqueeze(-1),
                       -w[:, 0, 1].unsqueeze(-1)], dim=1)
    else:
        zero1 = torch.zeros(n, 1, 1).to(device)
        # zero1 = torch.zeros(n, 1, 1)
        w = w.unsqueeze(-1).unsqueeze(-1)
        W = torch.cat([torch.cat([zero1, -w[:, 2], w[:, 1]], dim=2),
                       torch.cat([w[:, 2], zero1, -w[:, 0]], dim=2),
                       torch.cat([-w[:, 1], w[:, 0], zero1], dim=2)], dim=1)
    return W

def exp_so3(Input):
    device = get_device_info(Input)
    n = Input.shape[0]
    if Input.shape == (n, 3, 3):
        W = Input
        w = skew(Input)
    else:
        w = Input
        W = skew(w)

    wnorm_sq = torch.sum(w * w, dim=1)
    wnorm_sq_unsqueezed = wnorm_sq.unsqueeze(-1).unsqueeze(-1)

    wnorm = torch.sqrt(wnorm_sq)
    wnorm_unsqueezed = torch.sqrt(wnorm_sq_unsqueezed)

    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)
    w0 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
    w1 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
    w2 = w[:, 2].unsqueeze(-1).unsqueeze(-1)
    eps = 1e-7

    R = torch.zeros(n, 3, 3).to(device)

    R[wnorm > eps] = torch.cat((torch.cat((cw - ((w0 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
                                           - (w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           (w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2),
                                torch.cat(((w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           cw - ((w1 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
                                           - (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2),
                                torch.cat((-(w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
                                           cw - ((w2 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed),
                                          dim=2)),
                               dim=1)[wnorm > eps]

    R[wnorm <= eps] = torch.eye(3).to(device) + W[wnorm < eps] + 1 / 2 * W[wnorm < eps] @ W[wnorm < eps]
    return R

def exp_se3(S):
    device = get_device_info(S)
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3
    wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1,1)
    wnorm = torch.sqrt(wsqr)  # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  # dim = (n,1,1)
    wnorm_inv = 1 / wnorm_unsqueezed  # dim = (n)
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)

    eps = 1e-014
    W = skew(w)
    P = torch.eye(3, device=device) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    # P = torch.eye(3) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    P[wnorm < eps] = torch.eye(3, device=device)
    # P[wnorm < eps] = torch.eye(3)
    T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4, device=device))], dim=1)
    # T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4))], dim=1)
    T[:, -1, -1] = 1
    return T

def get_SE3s_torch(Rs, ps):
    assert Rs.ndim == ps.ndim + 1, f"Dimension of positions must be {Rs.ndim-1} if dimension of matrices is {Rs.ndim}"

    if Rs.ndim == 2:
        SE3s = torch.eye(4).to(Rs)
        SE3s[:3, :3] = Rs
        SE3s[:3, 3] = ps
    elif Rs.ndim == 3:
        SE3s = torch.cat([torch.eye(4).unsqueeze(0)] * len(Rs)).to(Rs)
        SE3s[:, :3, :3] = Rs
        SE3s[:, :3, 3] = ps
    else:
        raise NotImplementedError("Dimension of matrices must be 2 or 3")
    
    return SE3s
