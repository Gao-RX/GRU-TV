import torch
import torch.nn as nn

__all__ = ['GRUTCell', 'GRUTVCell']

class GRUTCell(nn.Module):
  def __init__(self, input_size, hidden_size, gamma_x, gamma_h):
    super().__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.gamma_x = gamma_x
    self.gamma_h = gamma_h
    self.mean_value = None
    self.inputzeros = torch.autograd.Variable(torch.zeros(input_size)).cuda()
    self.hiddenzeros = torch.autograd.Variable(torch.zeros(hidden_size)).cuda()
    
    self.w_dg_x = nn.Parameter(torch.Tensor(input_size))
    self.w_dg_h = nn.Parameter(torch.Tensor(hidden_size,input_size))
    self.b_dg_x = nn.Parameter(torch.Tensor(input_size))
    self.b_dg_h = nn.Parameter(torch.Tensor(hidden_size))

    self.lin_xh = nn.Linear(input_size, hidden_size, bias=True)
    self.lin_xz = nn.Linear(input_size, hidden_size, bias=True)
    self.lin_xr = nn.Linear(input_size, hidden_size, bias=True)
    self.lin_hu = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_mu = nn.Linear(input_size, hidden_size, bias=False)
    self.lin_mz = nn.Linear(input_size, hidden_size, bias=False)
    self.lin_mr = nn.Linear(input_size, hidden_size, bias=False)

  def forward(self, h, x, m, d, prex, dh):
    if self.gamma_x:
        gamma_x = torch.exp(-torch.max(self.inputzeros, (self.w_dg_x*d + self.b_dg_x)))
        x = m * x + (1 - m) * (gamma_x * prex + (1 - gamma_x) * self.mean_value)
    else:
        x = m * x + (1 - m) * prex
    if self.gamma_h:
        gamma_h = torch.exp(-torch.max(self.hiddenzeros, (torch.matmul(self.w_dg_h, d) + self.b_dg_h)))
        h = gamma_h * h
    r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h) + self.lin_mr(m))
    z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h) + self.lin_mz(m))
    u = torch.tanh(self.lin_xh(x) + self.lin_hu(r * h) + self.lin_mu(m))

    h_post = (1-z) * h + z * u
    dh = z * (u - h)

    return h_post, dh, x


class GRUTVCell(nn.Module):
  def __init__(self, input_size, hidden_size, gamma_x, gamma_h):
    super().__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.mean_value = None
    self.gamma_x = gamma_x
    self.gamma_h = gamma_h
    
    self.inputzeros = torch.autograd.Variable(torch.zeros(input_size)).cuda()
    self.hiddenzeros = torch.autograd.Variable(torch.zeros(hidden_size)).cuda()
    
    self.w_dg_x = nn.Parameter(torch.Tensor(input_size))
    self.w_dg_h = nn.Parameter(torch.Tensor(hidden_size,input_size))
    self.b_dg_x = nn.Parameter(torch.Tensor(input_size))
    self.b_dg_h = nn.Parameter(torch.Tensor(hidden_size))

    self.lin_xh = nn.Linear(input_size, hidden_size, bias=True)
    self.lin_xz = nn.Linear(input_size, hidden_size, bias=True)
    self.lin_xr = nn.Linear(input_size, hidden_size, bias=True)
    self.lin_hu = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_su = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_sz = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_sr = nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_mu = nn.Linear(input_size, hidden_size, bias=False)
    self.lin_mz = nn.Linear(input_size, hidden_size, bias=False)
    self.lin_mr = nn.Linear(input_size, hidden_size, bias=False)

  def forward(self, h, x, m, d, prex, dh):
    if self.gamma_x:
        gamma_x = torch.exp(-torch.max(self.inputzeros, (self.w_dg_x*d + self.b_dg_x)))
        x = m * x + (1 - m) * (gamma_x * prex + (1 - gamma_x) * self.mean_value)
    else:
        x = m * x + (1 - m) * prex
    if self.gamma_h:
        gamma_h = torch.exp(-torch.max(self.hiddenzeros, (torch.matmul(self.w_dg_h, d) + self.b_dg_h)))
        h = gamma_h * h
    r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h) + self.lin_sr(dh) +self.lin_mr(m))
    z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h) + self.lin_sz(dh) + self.lin_mz(m))
    u = torch.tanh(self.lin_xh(x) + self.lin_hu(r * h) + self.lin_su(dh) + self.lin_mu(m))

    h_post = (1-z) * h + z * u
    dh = z * (u - h)

    return h_post, dh, x

