"""Ribbon Synapse and Post-Synaptic Parallel Neurons.

リボンシナプス部：
1. 【前細胞のリボン構造（伝達物質小胞プール）】
   - 在庫プール（R）を持ち、音の刺激（エネルギー）に応じて小胞を放出。
   - 大音量が続くとプールが枯渇し、自然な順応（疲労・反動）が起きる。
   - 常に補充（Replenishment）されて在庫が復活する。

2. 【10〜20個の並列シナプス後細胞（聴神経線維）】
   - 閾値のグラデーション：かすかな音に反応する高感度細胞から、大音量でしか反応しない細胞まで配置。
   - 交代発火（ボレー原理）：不応期（発火後の休憩）により、細胞同士が交代で波を捉え、音の連続性をキープ。
   - 自発ノイズ：完全な無音時でもパチパチ鳴る背景ノイズ（確率共鳴で微小音をアシスト）。
"""

from typing import Tuple, Dict
import numpy as np
from config import RibbonSynapseConfig


class RibbonSynapseChannel:
    """単一の周波数チャネルにぶら下がる1つのリボンシナプス＋並列後細胞群"""
    
    def __init__(self, cfg: RibbonSynapseConfig):
        self.cfg = cfg
        N = cfg.num_post_neurons
        
        # 1. 閾値のグラデーション配置（低音量用〜大音量用）
        if cfg.threshold_distribution == "geometric":
            self.thresholds = np.geomspace(cfg.min_threshold, cfg.max_threshold, N)
        else:
            self.thresholds = np.linspace(cfg.min_threshold, cfg.max_threshold, N)
            
        # 2. 内部状態の初期化
        self.pool = cfg.pool_max                           # 伝達物質プール在庫
        self.voltages = np.zeros(N)                        # 後細胞の膜電位
        self.refractory_counters = np.zeros(N, dtype=int)  # 各細胞の不応期カウンタ
        
    def reset(self):
        """内部状態のリセット"""
        self.pool = self.cfg.pool_max
        self.voltages.fill(0.0)
        self.refractory_counters.fill(0)

    def step(self, energy: float) -> Tuple[np.ndarray, float]:
        """1ステップの計算。
        
        Args:
            energy: このチャネルに入力された受容音エネルギー (非負)
            
        Returns:
            spikes: 各後細胞のスパイク有無 (0 or 1, 長さ N)
            pool_level: 現在の伝達物質プールの残量 (0.0〜1.0)
        """
        N = self.cfg.num_post_neurons
        
        # ----------------------------------------------------
        # 1. 前細胞（リボン）からの伝達物質放出と補充
        # ----------------------------------------------------
        # プールの自動補充
        self.pool += self.cfg.replenish_rate * (self.cfg.pool_max - self.pool)
        
        # 音刺激に応じた放出量（在庫が減ると放出量も落ちる＝自然な枯渇と順応）
        # エネルギーが大きいほど放出率が上がる
        release_fraction = np.clip(self.cfg.release_rate * energy, 0.0, 1.0)
        released_neurotransmitter = self.pool * release_fraction
        
        # 在庫から引き落とし
        self.pool -= released_neurotransmitter
        
        # ----------------------------------------------------
        # 2. 10〜20個のシナプス後細胞への分配とスパイク生成
        # ----------------------------------------------------
        spikes = np.zeros(N, dtype=np.int8)
        
        # 常時聞こえるノイズ（自発発火用の背景ガウスノイズ）
        noise = np.random.normal(0.0, self.cfg.spontaneous_noise, size=N)
        
        for k in range(N):
            # 不応期中なら休憩（膜電位はリセット維持）
            if self.refractory_counters[k] > 0:
                self.refractory_counters[k] -= 1
                self.voltages[k] = 0.0
                continue
                
            # 膜電位のリーク積分 ＋ シナプス入力 ＋ ノイズ
            self.voltages[k] = (
                self.cfg.leak_rate * self.voltages[k]
                + released_neurotransmitter
                + noise[k]
            )
            
            # 閾値判定
            if self.voltages[k] >= self.thresholds[k]:
                spikes[k] = 1
                self.voltages[k] = 0.0  # リセット
                self.refractory_counters[k] = self.cfg.refractory_steps  # 不応期に入る
                
        return spikes, self.pool


class RibbonSynapseBank:
    """全周波数チャネル分のリボンシナプス集団"""
    
    def __init__(self, num_channels: int, cfg: RibbonSynapseConfig):
        self.num_channels = num_channels
        self.cfg = cfg
        self.channels = [RibbonSynapseChannel(cfg) for _ in range(num_channels)]
        
    def reset(self):
        for ch in self.channels:
            ch.reset()
            
    def process(self, energies: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """全チャネルの受容エネルギー時系列からスパイク列を生成する。
        
        Args:
            energies: (時間ステップ数, チャネル数)
            
        Returns:
            all_spikes: (時間ステップ数, チャネル数, 後細胞数)
            info: プール残量などの内部情報
        """
        num_steps, num_channels = energies.shape
        N = self.cfg.num_post_neurons
        
        all_spikes = np.zeros((num_steps, num_channels, N), dtype=np.int8)
        pool_history = np.zeros((num_steps, num_channels))
        
        for t in range(num_steps):
            for ch in range(num_channels):
                spikes, pool = self.channels[ch].step(energies[t, ch])
                all_spikes[t, ch, :] = spikes
                pool_history[t, ch] = pool
                
        info = {
            "pool_history": pool_history,
            "thresholds": self.channels[0].thresholds,  # 代表の閾値
        }
        return all_spikes, info
