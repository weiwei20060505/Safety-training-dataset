"""
模型內建條件分流分類器 (In-Model Conditional Branching Classifiers)
============================================================
本模組提供兩種 3 ~ 6 層切分與多頭分類器，用於根據 y1 (模型回覆安全性 0/1) 條件預測 y2 (提示詞有害性)：

1. RootSplitLGBMClassifier: LightGBM 樹狀模型，強迫根節點 (Root Node) 優先對 y1 進行切割。
2. YHeadMLPPyTorchClassifier: PyTorch 多頭 Y 字型神經網路，前兩層共用隱藏層，最後分拆為 Head 0 與 Head 1。
"""

import os
import json
import tempfile
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RootSplitLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    LightGBM 樹狀強迫分流分類器 (3 ~ 6 層深度)
    
    將 y1 作為第 0 個特徵插入特徵矩陣 [y1, X]，利用 forcedsplits_filename 強迫所有決策樹
    在 Root Node (Depth 0) 必須依據 y1 <= 0.5 進行切分。
    """
    def __init__(self, max_depth=5, num_leaves=31, n_estimators=100, learning_rate=0.05, random_state=42):
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"

    def _create_forced_splits_file(self):
        """生成強迫根節點以 feature 0 (y1) 切割的 JSON 設定檔"""
        forced_splits_dict = {
            "feature": 0,
            "threshold": 0.5,
            "left": {},
            "right": {}
        }
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(forced_splits_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def fit(self, X, y1, y2):
        """
        訓練 LightGBM 強迫切分模型
        Parameters:
            X: (N, D) 特徵矩陣
            y1: (N,) 條件 control 標籤 (0 或 1)
            y2: (N,) 目標標籤 (0 或 1)
        """
        X = np.asarray(X, dtype=np.float32)
        y1 = np.asarray(y1, dtype=np.float32).reshape(-1, 1)
        y2 = np.asarray(y2, dtype=int)

        # 拼接 [y1, X] => y1 為 Column 0
        X_all = np.hstack([y1, X])

        forced_json_path = self._create_forced_splits_file()
        try:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'max_depth': self.max_depth,
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'random_state': self.random_state,
                'forcedsplits_filename': forced_json_path,
                'min_data_in_leaf': 1,
                'min_child_samples': 1,
                'verbosity': -1,
                'is_unbalance': True
            }
            dtrain = lgb.Dataset(X_all, label=y2, categorical_feature=[0])
            self.model_ = lgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators
            )
        finally:
            if os.path.exists(forced_json_path):
                os.remove(forced_json_path)

        return self

    def predict_proba(self, X, y1):
        """根據給定的 (X, y1) 輸出預測機率 P(y2=1 | X, y1)"""
        X = np.asarray(X, dtype=np.float32)
        y1 = np.asarray(y1, dtype=np.float32).reshape(-1, 1)
        X_all = np.hstack([y1, X])
        raw_preds = self.model_.predict(X_all) # shape (N,)
        proba_1 = raw_preds.reshape(-1, 1)
        proba_0 = 1.0 - proba_1
        return np.hstack([proba_0, proba_1])

    def predict_proba_head0(self, X):
        """強迫設定 y1=0 (Safe 分支)，獲取 P(y2=1 | X, y1=0)"""
        y1_zero = np.zeros(len(X))
        return self.predict_proba(X, y1_zero)

        def predict_proba_head1(self, X):
            """強迫設定 y1=1 (Unsafe 分支)，獲取 P(y2=1 | X, y1=1)"""
            y1_one = np.ones(len(X))
            return self.predict_proba(X, y1_one)


class Feature129LGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    LightGBM 129維特徵全域探索分類器 (策略 2)
    將 y1 作為第 129 維度 (Column 0) 與 128 維隱藏狀態併入標準 LGB 模型，
    不加強迫切割，由資訊增益 (Information Gain) 自由尋找 y1 與特徵之切分點。
    """
    def __init__(self, max_depth=5, num_leaves=31, n_estimators=100, learning_rate=0.05, random_state=42):
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"

    def fit(self, X, y1, y2):
        X = np.asarray(X, dtype=np.float32)
        y1 = np.asarray(y1, dtype=np.float32).reshape(-1, 1)
        y2 = np.asarray(y2, dtype=int)
        X_all = np.hstack([y1, X])

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'min_data_in_leaf': 1,
            'min_child_samples': 1,
            'verbosity': -1,
            'is_unbalance': False
        }
        dtrain = lgb.Dataset(X_all, label=y2, categorical_feature=[0])
        self.model_ = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators
        )
        return self

    def predict_proba(self, X, y1=None):
        X = np.asarray(X, dtype=np.float32)
        if y1 is None:
            y1 = np.zeros(len(X))
        y1 = np.asarray(y1, dtype=np.float32).reshape(-1, 1)
        X_all = np.hstack([y1, X])
        raw_preds = self.model_.predict(X_all)
        proba_1 = raw_preds.reshape(-1, 1)
        proba_0 = 1.0 - proba_1
        return np.hstack([proba_0, proba_1])


if TORCH_AVAILABLE:
    class YHeadMLPNet(nn.Module):
        """
        Y字型多頭神經網路 (PyTorch)
        包含 2 層共用隱藏骨幹 + 各 2 層分支標頭 (Head 0 / Head 1)，總深度 4 層
        """
        def __init__(self, input_dim=128, shared_hidden=[256, 128], head_hidden=64, dropout_rate=0.2):
            super().__init__()
            
            # Shared Backbone (Layer 1 & Layer 2)
            self.shared_layer1 = nn.Sequential(
                nn.Linear(input_dim, shared_hidden[0]),
                nn.BatchNorm1d(shared_hidden[0]),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.shared_layer2 = nn.Sequential(
                nn.Linear(shared_hidden[0], shared_hidden[1]),
                nn.BatchNorm1d(shared_hidden[1]),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            
            # Head 0 (y1 = 0: Safe Branch, Layer 3 & Layer 4)
            self.head0 = nn.Sequential(
                nn.Linear(shared_hidden[1], head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1)
            )
            
            # Head 1 (y1 = 1: Unsafe Branch, Layer 3 & Layer 4)
            self.head1 = nn.Sequential(
                nn.Linear(shared_hidden[1], head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1)
            )

        def forward(self, x):
            feat = self.shared_layer1(x)
            feat = self.shared_layer2(feat)
            
            logit0 = self.head0(feat).squeeze(-1) # (N,)
            logit1 = self.head1(feat).squeeze(-1) # (N,)
            return logit0, logit1


    class SingleHead129MLPNet(nn.Module):
        """
        129維單頭標準神經網路 (PyTorch 策略 4)
        Input 129 -> 256 -> 128 -> 64 -> 1
        """
        def __init__(self, input_dim=129, hidden_dims=[256, 128, 64], dropout_rate=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.GELU(),
                nn.Linear(hidden_dims[2], 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


    class YHeadMLPPyTorchClassifier(BaseEstimator, ClassifierMixin):
        """
        PyTorch 實現之 Y 字型多頭條件分類器
        """
        def __init__(self, input_dim=128, epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4, random_state=42):
            self.input_dim = input_dim
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.model_ = None
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.classes_ = np.array([0, 1])
            self._estimator_type = "classifier"

        def fit(self, X, y1, y2, verbose=False):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y1_tensor = torch.tensor(y1, dtype=torch.float32)
            y2_tensor = torch.tensor(y2, dtype=torch.float32)

            dataset = TensorDataset(X_tensor, y1_tensor, y2_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model_ = YHeadMLPNet(input_dim=self.input_dim).to(self.device_)
            optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            bce_loss = nn.BCEWithLogitsLoss()

            self.model_.train()
            for epoch in range(self.epochs):
                total_loss = 0.0
                for bx, by1, by2 in dataloader:
                    bx, by1, by2 = bx.to(self.device_), by1.to(self.device_), by2.to(self.device_)
                    
                    optimizer.zero_grad()
                    logit0, logit1 = self.model_(bx)
                    
                    # Masked loss for Head 0 and Head 1
                    mask0 = (by1 == 0)
                    mask1 = (by1 == 1)
                    
                    loss0 = bce_loss(logit0[mask0], by2[mask0]) if mask0.any() else torch.tensor(0.0, device=self.device_)
                    loss1 = bce_loss(logit1[mask1], by2[mask1]) if mask1.any() else torch.tensor(0.0, device=self.device_)
                    
                    loss = loss0 + loss1
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(bx)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss / len(dataset):.4f}")

            return self

        def predict_proba(self, X, y1=None):
            """
            預測機率：若提供 y1，則對應選擇 Head 0 (y1=0) 或 Head 1 (y1=1) 之輸出。
            若未提供 y1，回傳預設為 Head 0 之輸出。
            """
            self.model_.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
            with torch.no_grad():
                logit0, logit1 = self.model_(X_tensor)
                p0 = torch.sigmoid(logit0).cpu().numpy()
                p1 = torch.sigmoid(logit1).cpu().numpy()

            if y1 is not None:
                y1 = np.asarray(y1)
                p_out = np.where(y1 == 1, p1, p0)
            else:
                p_out = p0

            proba_1 = p_out.reshape(-1, 1)
            proba_0 = 1.0 - proba_1
            return np.hstack([proba_0, proba_1])

        def predict_proba_head0(self, X):
            """單獨獲取 Head 0 (y1=0) 之預測機率 P(y2=1 | X, y1=0)"""
            y1_zero = np.zeros(len(X))
            return self.predict_proba(X, y1_zero)

        def predict_proba_head1(self, X):
            """單獨獲取 Head 1 (y1=1) 之預測機率 P(y2=1 | X, y1=1)"""
            y1_one = np.ones(len(X))
            return self.predict_proba(X, y1_one)


    class SingleHead129MLPPyTorchClassifier(BaseEstimator, ClassifierMixin):
        """
        PyTorch 129維單頭標準神經網路分類器 (策略 4)
        """
        def __init__(self, input_dim=128, epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4, random_state=42):
            self.input_dim = input_dim
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.model_ = None
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.classes_ = np.array([0, 1])
            self._estimator_type = "classifier"

        def fit(self, X, y1, y2, verbose=False):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
            X = np.asarray(X, dtype=np.float32)
            y1 = np.asarray(y1, dtype=np.float32).reshape(-1, 1)
            X_all = np.hstack([y1, X])
            
            X_tensor = torch.tensor(X_all, dtype=torch.float32)
            y2_tensor = torch.tensor(y2, dtype=torch.float32)

            dataset = TensorDataset(X_tensor, y2_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model_ = SingleHead129MLPNet(input_dim=self.input_dim + 1).to(self.device_)
            optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            bce_loss = nn.BCEWithLogitsLoss()

            self.model_.train()
            for epoch in range(self.epochs):
                total_loss = 0.0
                for bx, by2 in dataloader:
                    bx, by2 = bx.to(self.device_), by2.to(self.device_)
                    optimizer.zero_grad()
                    logits = self.model_(bx)
                    loss = bce_loss(logits, by2)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(bx)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss / len(dataset):.4f}")

            return self

        def predict_proba(self, X, y1=None):
            self.model_.eval()
            X = np.asarray(X, dtype=np.float32)
            if y1 is None:
                y1 = np.zeros(len(X))
            y1 = np.asarray(y1, dtype=np.float32).reshape(-1, 1)
            X_all = np.hstack([y1, X])
            X_tensor = torch.tensor(X_all, dtype=torch.float32).to(self.device_)
            with torch.no_grad():
                logits = self.model_(X_tensor)
                p1 = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
            p0 = 1.0 - p1
            return np.hstack([p0, p1])
else:
    class YHeadMLPPyTorchClassifier(BaseEstimator, ClassifierMixin):
        """PyTorch 未安裝時之 Dummy 佔位類別"""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch (torch) 未安裝，無法使用 YHeadMLPPyTorchClassifier。請執行 `pip install torch` 進行安裝。")

    class SingleHead129MLPPyTorchClassifier(BaseEstimator, ClassifierMixin):
        """PyTorch 未安裝時之 Dummy 佔位類別"""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch (torch) 未安裝，無法使用 SingleHead129MLPPyTorchClassifier。請執行 `pip install torch` 進行安裝。")


