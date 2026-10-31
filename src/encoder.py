"""Ribbon SNN Audio Encoder Main Module.

受容部（有毛細胞の帯域伸縮）とリボンシナプス部（プールと並列後細胞）を統合した
完全なオーディオエンコーダです。
"""

from typing import Tuple, Dict, Optional
import numpy as np
from config import EncoderConfig
from receptor import RibbonReceptorBank
from ribbon_synapse import RibbonSynapseBank


class RibbonAudioEncoder:
    """Ribbon Synapse Inspired SNN Audio Encoder."""
    
    def __init__(self, cfg: Optional[EncoderConfig] = None):
        if cfg is None:
            cfg = EncoderConfig()
        self.cfg = cfg
        
        # コンポーネントの構築
        self.receptor_bank = RibbonReceptorBank(cfg.receptor)
        self.synapse_bank = RibbonSynapseBank(cfg.receptor.num_channels, cfg.synapse)
        
    def reset(self):
        """エンコーダの内部状態をリセット"""
        self.receptor_bank.reset()
        self.synapse_bank.reset()
        
    def encode(
        self,
        audio: np.ndarray,
        flatten_channels: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """音声波形をSNN用スパイク列にエンコードする。
        
        Args:
            audio: 1次元の音声信号配列 (サンプル数,)
            flatten_channels: Trueの場合、(Steps, Channels * PostNeurons) の2次元配列で返す。
                              Falseの場合、(Steps, Channels, PostNeurons) の3次元配列で返す。
                              
        Returns:
            spikes: スパイクバイナリ配列 (0 または 1)
            info: 解析・可視化用の辞書
                  - 'times': 各時間ステップの秒数
                  - 'center_freqs': 各周波数チャネルの中心周波数 (Hz)
                  - 'width_history': 各チャネルの帯域幅 (Hz) の推移
                  - 'activity_history': 各チャネルの活動度の推移
                  - 'pool_history': 各チャネルのリボンプール残量の推移
                  - 'energies': 各チャネルの受容エネルギー
        """
        # 1. 周波数受容 ＆ 帯域幅の伸縮
        energies, receptor_info = self.receptor_bank.process(audio)
        
        # 2. リボンシナプス小胞放出 ＆ 並列後細胞のスパイク生成
        spikes_3d, synapse_info = self.synapse_bank.process(energies)
        
        # 統合情報
        info = {
            **receptor_info,
            **synapse_info,
            "energies": energies,
            "spikes_3d": spikes_3d,
        }
        
        if flatten_channels:
            steps, chs, neurons = spikes_3d.shape
            spikes_out = spikes_3d.reshape(steps, chs * neurons)
        else:
            spikes_out = spikes_3d
            
        return spikes_out, info
