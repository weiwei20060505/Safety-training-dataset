"""
測試模型內建條件分流分類器 (RootSplitLGBMClassifier & YHeadMLPPyTorchClassifier)
"""

import numpy as np
from conditional_models import RootSplitLGBMClassifier, YHeadMLPPyTorchClassifier

def test_models():
    print("=== 生成測試數據 ===")
    N = 1000
    D = 128
    np.random.seed(42)
    X = np.random.randn(N, D).astype(np.float32)
    y1 = np.random.choice([0, 1], size=N)
    
    # 建立具有條件關聯的 y2 標籤
    # 若 y1=0, y2 與 X[:, 0] 正相關；若 y1=1, y2 與 X[:, 1] 正相關
    y2 = np.where(y1 == 0, (X[:, 0] > 0).astype(int), (X[:, 1] > 0).astype(int))

    print("\n--- 測試 1: RootSplitLGBMClassifier ---")
    lgb_clf = RootSplitLGBMClassifier(max_depth=4, num_leaves=15, n_estimators=50)
    lgb_clf.fit(X, y1, y2)
    print("LGBM Fit 成功!")
    
    proba_cond = lgb_clf.predict_proba(X, y1)
    proba_h0 = lgb_clf.predict_proba_head0(X)
    proba_h1 = lgb_clf.predict_proba_head1(X)
    print(f"LGBM Conditional Proba Shape: {proba_cond.shape}")
    print(f"LGBM Head 0 Proba (y1=0) Mean: {proba_h0[:, 1].mean():.4f}")
    print(f"LGBM Head 1 Proba (y1=1) Mean: {proba_h1[:, 1].mean():.4f}")

    print("\n--- 測試 2: YHeadMLPPyTorchClassifier ---")
    try:
        mlp_clf = YHeadMLPPyTorchClassifier(input_dim=D, epochs=10, batch_size=32, lr=1e-3)
        mlp_clf.fit(X, y1, y2, verbose=True)
        print("MLP PyTorch Fit 成功!")
        
        proba_mlp_cond = mlp_clf.predict_proba(X, y1)
        proba_mlp_h0 = mlp_clf.predict_proba_head0(X)
        proba_mlp_h1 = mlp_clf.predict_proba_head1(X)
        print(f"MLP Conditional Proba Shape: {proba_mlp_cond.shape}")
        print(f"MLP Head 0 Proba Mean: {proba_mlp_h0[:, 1].mean():.4f}")
        print(f"MLP Head 1 Proba Mean: {proba_mlp_h1[:, 1].mean():.4f}")
    except Exception as e:
        print(f"MLP PyTorch 測試遭遇問題: {e}")

if __name__ == "__main__":
    test_models()
