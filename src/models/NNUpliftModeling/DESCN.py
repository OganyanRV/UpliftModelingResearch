class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, cfg, device):
        super(ShareNetwork, self).__init__()
        if cfg.get('BatchNorm1d', 'false') == 'true':
            self.DNN = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.get('do_rate', 0.2)),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.get('do_rate', 0.2)),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.get('do_rate', 0.2))
            )
        else:
            self.DNN = nn.Sequential(
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.get('do_rate', 0.2)),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.get('do_rate', 0.2)),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
            )

        self.DNN.apply(init_weights)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        h_rep = self.DNN(x)
        if self.cfg.get('normalization', 'none') == "divide":
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep_norm


class BaseModel(nn.Module):
    def __init__(self, base_dim, cfg):
        super(BaseModel, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.get('do_rate', 0.2)),
            nn.Linear(base_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.get('do_rate', 0.2)),
            nn.Linear(base_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.get('do_rate', 0.2))
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
    def __init__(self, input_dim, share_dim, base_dim, do_rate, device, batch_norm=False, normalization="none"):
        super(DESCN, self).__init__()
        # Конфигурация модели
        cfg = {
            'do_rate': do_rate,
            'BatchNorm1d': 'true' if batch_norm else 'false',
            'normalization': normalization
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

    def forward(self, inputs):
        shared_h = self.shareNetwork(inputs)

        # propensity output_logit
        p_prpsy_logit = self.prpsy_network(shared_h)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1  # Refer to the naming in TARnet/CFR
        p_h0 = p_mu0  # Refer to the naming in TARnet/CFR

        # entire space
        p_estr = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)
        
        # Рассчитываем аплифт (эффект воздействия)
        uplift = mu1_logit - mu0_logit

        return {
            'p_prpsy_logit': p_prpsy_logit,
            'p_estr': p_estr,
            'p_escr': p_escr,
            'tau_logit': tau_logit,
            'mu1_logit': mu1_logit,
            'mu0_logit': mu0_logit,
            'p_prpsy': p_prpsy,
            'p_mu1': p_mu1,
            'p_mu0': p_mu0,
            'p_h1': p_h1,
            'p_h0': p_h0,
            'shared_h': shared_h,
            'uplift': uplift,
            'y1': mu1_logit,
            'y0': mu0_logit
        }


class DESCNUplift(NNUpliftModeling):
    """
    Реализация модели DESCN для аплифт-моделирования.
    """
    
    def _initialize_model(self):
        """
        Инициализация архитектуры модели DESCN.
        """
        input_dim = self.config.get('input_dim')
        share_dim = self.config.get('share_dim', 128)
        base_dim = self.config.get('base_dim', 64)
        do_rate = self.config.get('do_rate', 0.2)
        batch_norm = self.config.get('batch_norm', False)
        normalization = self.config.get('normalization', 'none')
        
        # Проверка наличия обязательных параметров
        if input_dim is None:
            raise ValueError("input_dim must be specified in the config")
        
        # Инициализация модели
        self.model = DESCN(
            input_dim=input_dim,
            share_dim=share_dim,
            base_dim=base_dim,
            do_rate=do_rate,
            device=self.device,
            batch_norm=batch_norm,
            normalization=normalization
        )
    
    def _compute_loss(self, outputs, outcome, treatment):
        """
        Вычисление функции потерь для DESCN.
        
        Args:
            outputs: выход модели
            outcome: целевая переменная
            treatment: индикатор воздействия
            
        Returns:
            Значение функции потерь
        """
        # Извлечение необходимых выходов модели
        mu1_logit = outputs['mu1_logit']
        mu0_logit = outputs['mu0_logit']
        p_prpsy_logit = outputs['p_prpsy_logit']
        
        # Веса для разных компонентов потери
        factual_loss_weight = self.config.get('factual_loss_weight', 1.0)
        propensity_loss_weight = self.config.get('propensity_loss_weight', 0.1)
        tau_loss_weight = self.config.get('tau_loss_weight', 0.1)
        
        # Формируем маски для групп воздействия и контроля
        treatment_mask = (treatment == 1).float().unsqueeze(1)
        control_mask = (treatment == 0).float().unsqueeze(1)
        
        # Фактическая потеря - MSE для фактических наблюдений
        y_pred = treatment_mask * mu1_logit + control_mask * mu0_logit
        factual_loss = F.mse_loss(y_pred, outcome.unsqueeze(1))
        
        # Потеря для предсказания вероятности назначения воздействия
        propensity_loss = F.binary_cross_entropy_with_logits(
            p_prpsy_logit.squeeze(), 
            treatment
        )
        
        # Потеря для предсказания эффекта воздействия (если известно)
        if self.config.get('use_tau_loss', False) and hasattr(self, 'tau_true'):
            tau_loss = F.mse_loss(outputs['tau_logit'], self.tau_true)
        else:
            tau_loss = torch.tensor(0.0, device=self.device)
        
        # Общая потеря
        total_loss = (
            factual_loss_weight * factual_loss + 
            propensity_loss_weight * propensity_loss + 
            tau_loss_weight * tau_loss
        )
        
        return total_loss
    
    def _process_prediction_outputs(self, outputs):
        """
        Обработка выходов модели для предсказания.
        
        Args:
            outputs: выходы модели
            
        Returns:
            Словарь с предсказанными значениями
        """
        # Выделяем и преобразуем нужные для предсказания поля
        return {
            'y0': outputs['mu0_logit'],
            'y1': outputs['mu1_logit'],
            'uplift': outputs['uplift'],
            'propensity': outputs['p_prpsy']
        }
    
    @staticmethod
    def generate_config(count, **params):
        """
        Генерация конфигураций для DESCN модели.
        
        Args:
            count: количество конфигураций
            **params: дополнительные параметры
            
        Returns:
            Список конфигураций
        """
        # Базовые параметры для DESCN
        descn_params = {
            'input_dim': 32,             # Должно быть задано в соответствии с данными
            'share_dim': [64, 128, 256], # Варианты размерности общих слоев
            'base_dim': [32, 64, 128],   # Варианты размерности базовых слоев
            'do_rate': [0.1, 0.2, 0.3],  # Варианты dropout
            'batch_norm': [True, False], # Использование BatchNorm
            'normalization': ['none', 'divide'], # Тип нормализации
            'factual_loss_weight': [0.8, 1.0, 1.2], # Вес фактической потери
            'propensity_loss_weight': [0.05, 0.1, 0.2], # Вес потери пропенсити
            'tau_loss_weight': [0.05, 0.1, 0.2]    # Вес потери tau (если применимо)
        }
        
        # Объединение с переданными параметрами
        for key, value in params.items():
            descn_params[key] = value
        
        # Генерация конфигураций с использованием базового метода
        return NNUpliftModeling.generate_config(count, **descn_params)