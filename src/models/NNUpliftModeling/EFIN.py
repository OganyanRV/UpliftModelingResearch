import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class EFIN(nn.Module):
    """
    EFIN (Explicit Feature Interaction Network) для аплифт-моделирования.
    """
    def __init__(self, input_dim, hc_dim, hu_dim, act_type='elu', version='3'):
        """
        Инициализация модели EFIN.
        
        Args:
            input_dim: Количество входных признаков
            hc_dim: Размерность скрытых слоев control net
            hu_dim: Размерность скрытых слоев uplift net
        """
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        
        self.att_embed_1 = nn.Linear(hu_dim, hu_dim, bias=False)
        self.att_embed_2 = nn.Linear(hu_dim, hu_dim)
        self.att_embed_3 = nn.Linear(hu_dim, 1, bias=False)

        # self-attention
        self.softmax = nn.Softmax(dim=-1)
        self.Q_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.K_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.V_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)

        # Эмбеды признаков
        self.x_rep = nn.Embedding(input_dim, hu_dim)

        # Эмбеды тритмента
        self.t_rep = nn.Linear(1, hu_dim)

        # Control Net (для предсказания без воздействия)
        if version == '3':
            self.c_layers = nn.ModuleList([
                nn.Linear(input_dim * hu_dim, hc_dim),
                nn.Linear(hc_dim, hc_dim),
                nn.Linear(hc_dim, hc_dim // 2),
                nn.Linear(hc_dim // 2, hc_dim // 4)
            ])
            out_dim = hc_dim // 4
        else: 
            self.c_layers = nn.ModuleList([
                nn.Linear(input_dim * hu_dim, hc_dim),
                nn.Linear(hc_dim, hc_dim),
                nn.Linear(hc_dim, hc_dim // 2),
                nn.Linear(hc_dim // 2, hc_dim // 2),
                nn.Linear(hc_dim // 2, hc_dim // 4),
                nn.Linear(hc_dim // 4, hc_dim // 4)
            ])
            out_dim = hc_dim // 4

        self.c_logit = nn.Linear(out_dim, 1)
        self.c_tau = nn.Linear(out_dim, 1)

        # Uplift Net (для моделирования инкрементального эффекта)
        if version == '3':
            self.u_layers = nn.ModuleList([
                nn.Linear(hu_dim, hu_dim),
                nn.Linear(hu_dim, hu_dim // 2),
                nn.Linear(hu_dim // 2, hu_dim // 4)
            ])
            out_dim = hu_dim // 4
        else:
            self.u_layers = nn.ModuleList([
                nn.Linear(hu_dim, hu_dim),
                nn.Linear(hu_dim, hu_dim // 2),
                nn.Linear(hu_dim // 2, hu_dim // 2),
                nn.Linear(hu_dim // 2, hu_dim // 4),
                nn.Linear(hu_dim // 4, hu_dim // 4)
            ])
            out_dim = hu_dim // 4
        
        self.t_logit = nn.Linear(out_dim, 1)
        self.u_tau = nn.Linear(out_dim, 1)

        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError(f'Неизвестный тип активации: {act_type}')

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))

        outputs = attn_weights.matmul(V)

        return outputs, attn_weights

    def interaction_attn(self, t, x):
        attention = []
        for i in range(self.nums_feature):
            temp = self.att_embed_3(torch.relu(
                torch.sigmoid(self.att_embed_1(t)) + torch.sigmoid(self.att_embed_2(x[:, i, :]))))
            attention.append(temp)
        attention = torch.squeeze(torch.stack(attention, 1), 2)
        attention = torch.softmax(attention, 1)

        outputs = torch.squeeze(torch.matmul(torch.unsqueeze(attention, 1), x), 1)
        return outputs, attention

    def forward(self, features, is_treat):
        """
        features: Входные признаки
        is_treat: Индикатор воздействия (1 = воздействие, 0 = контроль)
        """
        t_true = torch.unsqueeze(is_treat, 1)
        
        x_rep = features.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)

        # Control Net
        dims = x_rep.size()
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True)
        xx, xx_weight = self.self_attn(_x_rep, _x_rep, _x_rep)

        _x_rep = torch.reshape(xx, (dims[0], dims[1] * dims[2]))

        c_last = _x_rep
        for layer in self.c_layers:
            c_last = self.act(layer(c_last))
        
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # Uplift Net
        t_rep = self.t_rep(torch.ones_like(t_true))

        xt, xt_weight = self.interaction_attn(t_rep, x_rep)
        
        u_last = xt
        for layer in self.u_layers:
            u_last = self.act(layer(u_last))

        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)
        
        # For predictions
        c_logit_fix = c_logit.detach()
        uc = c_logit  # Предсказание без воздействия
        ut = (c_logit_fix + u_tau)  # Предсказание с воздействием

        outputs = {
            'c_logit': c_logit,   # Логиты для контрольной группы
            'c_prob': c_prob,     # Вероятности для контрольной группы
            'c_tau': c_tau,       # Tau для контрольной группы
            't_logit': t_logit,   # Логиты для группы воздействия
            't_prob': t_prob,     # Вероятности для группы воздействия
            'u_tau': u_tau,       # Tau для uplift
            'uc': uc,             # Logits для y0
            'ut': ut,             # Logits для y1
            'uplift': t_prob - c_prob,  # Предсказание аплифта (p1 - p0)
            'p_mu0': c_prob,       # Вероятность положительного исхода без воздействия
            'p_mu1': t_prob,       # Вероятность положительного исхода с воздействием
            'mu0_logit': c_logit,  # Логиты для исхода без воздействия
            'mu1_logit': t_logit   # Логиты для исхода с воздействием
        }
        
        return outputs
