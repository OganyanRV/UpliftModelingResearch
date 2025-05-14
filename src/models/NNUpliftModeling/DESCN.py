import torch
import torch.nn as nn
import math

def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def safe_sqrt(x):
    ''' Numerically safe version of Pytoch sqrt '''
    return torch.sqrt(torch.clip(x, 1e-9, 1e+9))


class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, cfg, device):
        super(ShareNetwork, self).__init__()
        
        self.cfg = cfg
        self.device = device
        descn_version = cfg.get('descn_version', '3')
        dropout_rate = cfg.get('do_rate', 0.2)
        use_batch_norm = cfg.get('BatchNorm1d', 'false') == 'true'
        
        # Создаем списки слоев для обоих версий
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        
        if descn_version == '3':
            # 3-слойная архитектура
            layers.extend([
                nn.Linear(input_dim, share_dim),  # layer 0
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(share_dim, share_dim),  # layer 1
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(share_dim, base_dim),   # layer 2
                nn.ELU()
            ])
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
        else:  # descn_version == '6'
            # 6-слойная архитектура
            layers.extend([
                nn.Linear(input_dim, share_dim),  # layer 0
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(share_dim, share_dim),  # layer 1
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(share_dim, share_dim),  # layer 2
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(share_dim, share_dim),  # layer 3
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(share_dim, share_dim),  # layer 4
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(share_dim, base_dim),   # layer 5
                nn.ELU()
            ])
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
        
        self.layers = nn.ModuleList(layers)
        self.layers.apply(init_weights)
        
        # Сохраняем индексы линейных слоев для удобства
        self.linear_indices = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Linear)]
        self.to(device)

    def forward(self, x, return_intermediates=False):
        x = x.to(self.device)
        
        # Собираем промежуточные выходы, если нужно
        intermediates = {}
        h = x
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            # Сохраняем выходы линейных слоев
            if return_intermediates and i in self.linear_indices:
                linear_idx = self.linear_indices.index(i)
                intermediates[f'layer_{linear_idx}'] = h
        
        # Применяем нормализацию
        if self.cfg.get('normalization', 'none') == "divide":
            h_norm = h / safe_sqrt(torch.sum(torch.square(h), dim=1, keepdim=True))
        else:
            h_norm = 1.0 * h
        
        if return_intermediates:
            return h_norm, intermediates
        else:
            return h_norm


class BaseModel(nn.Module):
    def __init__(self, base_dim, cfg):
        super(BaseModel, self).__init__()
        
        n_layers = cfg.get('descn_version', '3')
        dropout_rate = cfg.get('do_rate', 0.2)
        
        # Архитектура с 3 слоями
        if n_layers == '3':
            self.DNN = nn.Sequential(
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate)
            )
        
        else:
            self.DNN = nn.Sequential(
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(base_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=dropout_rate)
            )
            
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits


class PrpsyNetwork(nn.Module):
    """propensity network"""
    def __init__(self, base_dim, cfg):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, cfg):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, cfg):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class TauNetwork(nn.Module):
    """pseudo tau network"""
    def __init__(self, base_dim, cfg):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        return tau_logit


class DESCN(nn.Module):
    """DESCN (Deep End-to-end Stochastic Causal Network)"""
    def __init__(self, input_dim, share_dim, base_dim, do_rate, device, 
                 batch_norm=False, normalization="none", descn_version='3',
                 return_intermediates=False):
        super(DESCN, self).__init__()
        # Конфигурация модели
        cfg = {
            'do_rate': do_rate,
            'BatchNorm1d': 'true' if batch_norm else 'false',
            'normalization': normalization,
            'descn_version': descn_version
        }
        
        # Компоненты модели
        self.shareNetwork = ShareNetwork(input_dim, share_dim, base_dim, cfg, device)
        self.prpsy_network = PrpsyNetwork(base_dim, cfg)
        self.mu1_network = Mu1Network(base_dim, cfg)
        self.mu0_network = Mu0Network(base_dim, cfg)
        self.tau_network = TauNetwork(base_dim, cfg)
        
        self.cfg = cfg
        self.device = device
        self.to(device)
        
        self.return_intermediates = return_intermediates

    def forward(self, inputs):
        
        if self.return_intermediates:
            intermediates = {}
            shared_h, share_intermediates = self.shareNetwork(inputs, return_intermediates=True)
            intermediates['share'] = share_intermediates
        else:
            shared_h = self.shareNetwork(inputs)

        # propensity output_logit
        p_prpsy_logit = self.prpsy_network(shared_h)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = torch.sigmoid(mu1_logit)
        p_mu0 = torch.sigmoid(mu0_logit)

        # entire space
        p_estr = torch.mul(p_prpsy, p_mu1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_mu0)
        
        # Рассчитываем аплифт (эффект воздействия)
        uplift = p_mu1 - p_mu0
        # uplift = p_estr - p_escr

        outputs = {
            'p_prpsy_logit': p_prpsy_logit,
            'p_estr': p_estr,
            'p_escr': p_escr,
            'tau_logit': tau_logit,
            'mu1_logit': mu1_logit,
            'mu0_logit': mu0_logit,
            'p_prpsy': p_prpsy,
            'p_mu1': p_mu1,
            'p_mu0': p_mu0,
            'shared_h': shared_h,
            'uplift': uplift
        }
        
        # Добавляем промежуточные выходы, если нужно
        if self.return_intermediates:
            outputs['intermediates'] = intermediates
        
        return outputs