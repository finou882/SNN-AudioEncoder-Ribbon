"""Receptor (Cochlear Hair Cell with Dynamic Receptive Field Tuning)

周波数受容部：
- 低音から高音まで、隣り合うチャネルが適度にかぶり（オーバーラップ）を持つように対数スケールで配置。
- 入力音声の周波数エネルギーを感知し、活動度（反応頻度）をH&H風の時定数で追跡。
- 反応頻度に応じて、受容範囲（帯域幅）を動的に伸縮（敏感にピントを絞り、音が止んだらすぐ広がる）。
"""

from typing import Tuple, List, Dict
import numpy as np
from scipy import signal
from config import ReceptorConfig


class RibbonReceptorBank:
    """動的帯域伸縮フィルタバンク（有毛細胞群）"""
    
    def __init__(self, cfg: ReceptorConfig):
        self.cfg = cfg
        
        # 1. 中心周波数の配置（人間の耳と同じ対数スケール / ERBスケールに近い等比配置）
        self.center_freqs = np.geomspace(cfg.min_freq, cfg.max_freq, cfg.num_channels)
        
        # 2. 各チャネルの最小・最大帯域幅（Hz）
        self.min_widths = self.center_freqs * cfg.min_width_factor
        self.max_widths = self.center_freqs * cfg.max_width_factor
        
        # 3. 内部状態の初期化
        self.current_widths = self.max_widths.copy()  # 初期状態はアンテナ全開（最大幅）
        self.activities = np.zeros(cfg.num_channels)   # 活動度 (0.0〜1.0)
        
    def reset(self):
        """内部状態のリセット"""
        self.current_widths = self.max_widths.copy()
        self.activities.fill(0.0)

    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """音声信号を入力し、各チャネルの受容エネルギーと帯域幅履歴を計算する。
        
        Args:
            audio: 1次元の音声波形 (サンプル数,)
            
        Returns:
            energies: 各チャネルの受容エネルギー時系列 (時間ステップ数, チャネル数)
            info: デバッグ・可視化用の情報（帯域幅の軌跡、活動度の軌跡など）
        """
        sr = self.cfg.sample_rate
        hop_samples = int(sr * (self.cfg.hop_length_ms / 1000.0))
        n_fft = max(256, 1 << (hop_samples * 2 - 1).bit_length())  # 2のべき乗
        
        # 短時間フーリエ変換 (STFT) でスペクトログラムを抽出
        freqs, times, Zxx = signal.stft(
            audio,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop_samples,
            boundary=None
        )
        spec = np.abs(Zxx)  # 振幅スペクトル (周波数ビン数, 時間ステップ数)
        num_steps = spec.shape[1]
        
        # 記録用のバッファ
        energy_history = np.zeros((num_steps, self.cfg.num_channels))
        width_history = np.zeros((num_steps, self.cfg.num_channels))
        activity_history = np.zeros((num_steps, self.cfg.num_channels))
        
        # 時間ステップごとに動的に帯域幅を更新しながら受容エネルギーを算出
        for t in range(num_steps):
            frame_spectrum = spec[:, t]
            
            for ch in range(self.cfg.num_channels):
                fc = self.center_freqs[ch]
                w = self.current_widths[ch]
                
                # ガウシアン型フィルタ（現在の帯域幅 w を反映）
                # w は半値全幅(FWHM)に相当させるため sigma = w / 2.355
                sigma = max(1.0, w / 2.355)
                gaussian_kernel = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
                
                # フィルタを通過したエネルギー（内積）
                raw_response = np.sum(frame_spectrum * gaussian_kernel) / (np.sum(gaussian_kernel) + 1e-8)
                energy_history[t, ch] = raw_response
                
                # ① 活動度（反応頻度）の更新 (H&Hの時定数/リーク積分に相当)
                # 刺激の強さを 0〜1 程度に正規化して反映
                norm_response = np.clip(raw_response * 5.0, 0.0, 1.0)
                self.activities[ch] = (
                    self.cfg.activity_decay * self.activities[ch]
                    + (1.0 - self.cfg.activity_decay) * norm_response
                )
                
                # ② 帯域幅の伸縮（非対称な更新ルール）
                act = self.activities[ch]
                min_w = self.min_widths[ch]
                max_w = self.max_widths[ch]
                
                if act > self.cfg.narrow_threshold:
                    # 低い閾値を超えたら敏感にピントを絞る（狭まる）
                    self.current_widths[ch] -= self.cfg.shrink_rate * (self.current_widths[ch] - min_w)
                else:
                    # 音が止んだらすぐに広がる
                    self.current_widths[ch] += self.cfg.expand_rate * (max_w - self.current_widths[ch])
                
                # 安全のためのクリップ
                self.current_widths[ch] = np.clip(self.current_widths[ch], min_w, max_w)
                
                width_history[t, ch] = self.current_widths[ch]
                activity_history[t, ch] = self.activities[ch]
                
        info = {
            "center_freqs": self.center_freqs,
            "width_history": width_history,
            "activity_history": activity_history,
            "times": times,
        }
        return energy_history, info
