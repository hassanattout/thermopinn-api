import torch


def physics_loss(model, x, y):
    """
    Enforces heat equation:
        ∇²T + Q = 0
    """

    x.requires_grad_(True)
    y.requires_grad_(True)

    inputs = torch.cat([x, y], dim=1)
    T = model(inputs)

    # First derivatives
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    # Second derivatives
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]

    laplacian = T_xx + T_yy

    Q = 1000.0

    loss = torch.mean((laplacian + Q) ** 2)

    return loss
