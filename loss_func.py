import torch


class PinnLoss:
    def __init__(self):
        return

    def __call__(self, x, y, sigma_x, sigma_y, tau_xy, bc):
        pde_loss = self.cauchy_equilibrium(x, y, sigma_x, sigma_y, tau_xy)
        bc_loss = self.boundary_condition(sigma_x, sigma_y, tau_xy, bc)
        return pde_loss + bc_loss

    def cauchy_equilibrium(self, x, y, sigma_x, sigma_y, tau_xy):
        grad_outputs = torch.ones_like(x)

        dsigma_x_dx = torch.autograd.grad(sigma_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dy = torch.autograd.grad(tau_xy, y, grad_outputs=grad_outputs, create_graph=True)[0]
        eq1 = dsigma_x_dx + dtau_xy_dy

        dsigma_y_dy = torch.autograd.grad(sigma_y, y, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dx = torch.autograd.grad(tau_xy, x, grad_outputs=grad_outputs, create_graph=True)[0]
        eq2 = dtau_xy_dx + dsigma_y_dy
        return torch.sum(eq1 ** 2) + torch.sum(eq2 ** 2)

    def boundary_condition(self, sigma_x, sigma_y, tau_xy, bc):
        # Filter out the normal vectors and surface force components
        l = bc[:, :, 0:1]
        m = bc[:, :, 1:2]
        fx = bc[:, :, 2:3]
        fy = bc[:, :, 3:4]
        # Filter out the boundary points
        mask = bc[:, :, 4:5] == 1

        # Calculate loss only for the boundary points
        sigma_x_bc = sigma_x[mask]
        sigma_y_bc = sigma_y[mask]
        tau_xy_bc = tau_xy[mask]
        l_bc = l[mask]
        m_bc = m[mask]
        fx_bc = fx[mask]
        fy_bc = fy[mask]

        # Loss: (l * σ_x) + (m * τ_xy) = fx, (l * τ_xy) + (m * σ_y) = fy
        bc1 = (l_bc * sigma_x_bc) + (m_bc * tau_xy_bc) - fx_bc
        bc2 = (l_bc * tau_xy_bc) + (m_bc * sigma_y_bc) - fy_bc
        return torch.sum(bc1 ** 2) + torch.sum(bc2 ** 2)
