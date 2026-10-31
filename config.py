"""Configuration for Ribbon SNN Audio Encoder.

すべてのハイパーパラメータをここで一元管理し、手軽に調整・実験できるように設計されています。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReceptorConfig:
    """周波数受容部（有毛細胞の周波数選択性と帯域伸縮）の設定"""
    # 周波数配置
    sample_rate: int = 16000          # サンプリング周波数 (Hz)
    num_channels: int = 16            # 周波数チャネル数（有毛細胞の個数）
    min_freq: float = 200.0           # 最小周波数 (Hz)
    max_freq: float = 7000.0          # 最大周波数 (Hz)
    
    # 帯域幅（ピント）の伸縮設定
    min_width_factor: float = 0.08    # ピント絞った時の帯域幅（中心周波数に対する比率）
    max_width_factor: float = 0.35    # アンテナ広げた時の帯域幅（中心周波数に対する比率）
    
    # H&H風の活動度と帯域幅の更新ルール
    narrow_threshold: float = 0.15    # 低い閾値：これを超えるとピントを絞り始める
    shrink_rate: float = 0.05         # 絞る速度（敏感に狭まる）
    expand_rate: float = 0.12         # 広がる速度（音が止んだらすぐ広がる）
    activity_decay: float = 0.88      # 活動度の過去の記憶保持率 (H&Hの時定数に相当)
    
    # 時間・フレーム設定
    hop_length_ms: float = 2.0        # ステップ時間 (ms) - 1ステップあたりの更新間隔


@dataclass
class RibbonSynapseConfig:
    """リボンシナプス部（伝達物質プールと10~20個の並列後細胞）の設定"""
    # 後細胞（聴神経線維）の構成
    num_post_neurons: int = 15        # 1つの前細胞あたりの後細胞数 (10〜20)
    
    # 閾値のグラデーション（音の大きさを分担: Optuna 最適値）
    min_threshold: float = 0.19       # 最も敏感な細胞の閾値（かすかな微小音用）
    max_threshold: float = 0.61       # 最も鈍感な細胞の閾値（大音量用）
    threshold_distribution: str = "geometric"  # "linear" または "geometric" (対数/べき乗的配置)
    
    # 後細胞のニューロンダイナミクス (LIF: Optuna 最適値)
    leak_rate: float = 0.805          # 膜電位の保持率（リーク積分、時定数に相当）
    refractory_steps: int = 1         # 不応期（1ステップ=2ms、超高速追従とボレー交代発火用）
    
    # リボンシナプス小胞プール（音のアタックと順応・枯渇反動: Optuna 最適値）
    pool_max: float = 1.0             # プールの最大容量
    replenish_rate: float = 0.147     # プールの補充速度（順応後の回復を早め、次のイベントに即時スタンバイ）
    release_rate: float = 2.47        # 音刺激に対する放出率（アタック時の急峻な立ち上がりと高閾値動員）
    
    # 常時聞こえるノイズ（自発発火・確率共鳴: Optuna 最適値）
    spontaneous_noise: float = 0.021  # 膜電位に加わる背景ガウスノイズの標準偏差 (0で完全オフ)


@dataclass
class EncoderConfig:
    """エンコーダ全体の統合設定"""
    receptor: ReceptorConfig = field(default_factory=ReceptorConfig)
    synapse: RibbonSynapseConfig = field(default_factory=RibbonSynapseConfig)
