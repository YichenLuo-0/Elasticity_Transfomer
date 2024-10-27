import torch


class PinnLoss:
    def __init__(self, e):
        self.e = e
        return

    def __call__(self, x, y, u, v, sigma_x, sigma_y, tau_xy, bc, u_gt, v_gt, sigma_x_gt, sigma_y_gt, tau_xy_gt):
        # pde_loss = self.cauchy_equilibrium(x, y, sigma_x, sigma_y, tau_xy)
        # bc_loss = self.disp_boundary(u, v, bc) + self.force_boundary(sigma_x, sigma_y, tau_xy, bc)
        data_loss = self.data_loss(u, v, sigma_x, sigma_y, tau_xy, u_gt, v_gt, sigma_x_gt, sigma_y_gt, tau_xy_gt)
        return data_loss

    def data_loss(self, u, v, sigma_x, sigma_y, tau_xy, u_gt, v_gt, sigma_x_gt, sigma_y_gt, tau_xy_gt):
        # return torch.sum((u - u_gt) ** 2) + torch.sum((v - v_gt) ** 2) +
        return (torch.sum((sigma_x - sigma_x_gt) ** 2) + torch.sum((sigma_y - sigma_y_gt) ** 2)
                + torch.sum((tau_xy - tau_xy_gt) ** 2))

    def cauchy_equilibrium(self, x, y, sigma_x, sigma_y, tau_xy):
        grad_outputs = torch.ones_like(x)

        # dσ_x/dx + dτ_xy/dy = 0
        dsigma_x_dx = torch.autograd.grad(sigma_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dy = torch.autograd.grad(tau_xy, y, grad_outputs=grad_outputs, create_graph=True)[0]
        eq1 = dsigma_x_dx + dtau_xy_dy

        # dτ_xy/dx + dσ_y/dy = 0
        dsigma_y_dy = torch.autograd.grad(sigma_y, y, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dx = torch.autograd.grad(tau_xy, x, grad_outputs=grad_outputs, create_graph=True)[0]
        eq2 = dtau_xy_dx + dsigma_y_dy
        return torch.sum(eq1 ** 2) + torch.sum(eq2 ** 2)

    def disp_boundary(self, u, v, bc):
        us = bc[:, :, 4:5]
        vs = bc[:, :, 5:6]

        # Filter out the displacement boundary points
        mask = bc[:, :, 6:7] == 1
        # Calculate loss only for the boundary points
        u_bc = u[mask]
        v_bc = v[mask]
        us_bc = us[mask]
        vs_bc = vs[mask]

        # Loss: (u_bc - us_bc) * E, (v_bc - vs_bc) * E
        # Since the displacement boundary values are very small,
        # we need to multiply them by the Young's modulus to increase their influence
        bc1 = (u_bc - us_bc) * self.e
        bc2 = (v_bc - vs_bc) * self.e
        return torch.sum(bc1 ** 2) + torch.sum(bc2 ** 2)

    def force_boundary(self, sigma_x, sigma_y, tau_xy, bc):
        l = bc[:, :, 0:1]
        m = bc[:, :, 1:2]
        fx = bc[:, :, 2:3]
        fy = bc[:, :, 3:4]

        # Filter out the force boundary points
        mask = bc[:, :, 6:7] == 2
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
