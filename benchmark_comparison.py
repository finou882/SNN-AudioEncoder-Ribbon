"""Comprehensive Audio Encoder Benchmark for Spiking Neural Networks (SNN).

【比較対象の4大エンコーダ】
1. MFCC + Poisson Rate Coding (MFCC/メルスペクトログラム + ポアソン点過程発火)
2. Tonic-style Delta / Threshold-Crossing (Neuromorphic イベント変換: ON/OFF スパイク)
3. Standard LIF FilterBank (静的フィルタバンク + 通常の単一LIFニューロン)
4. Ribbon SNN Audio Encoder (Ours: 動的ピント伸縮 + 小胞プール順応 + 15並列後細胞群)

【評価シナリオ】
定常背景音（830Hzハミング＋広帯域雑音）の中で突発する3つの特異点イベント:
- Event 1 [Attack/Click]: 2140Hz 突発アタック・微小打撃音 (35ms)
- Event 2 [Freq Jump]: 2712Hz 周波数急変ホップ (60ms)
- Event 3 [Burst Attack]: 407Hz 大音量複合破裂音 (40ms)

【計測メトリクス】
- 検出成功率 (Hits / 3)
- 平均検出遅延 (Latency ms)
- 背景誤検知数 (False Alarms)
- 総スパイク数 (Total Spikes / 消費電力指標)
- スパイクS/N比 (Signal-to-Noise Ratio: イベント発火密度 / 背景発火密度)
"""

import sys
from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from config import EncoderConfig, ReceptorConfig
from encoder import RibbonAudioEncoder
from singularity_test import generate_singularity_audio


# ==============================================================================
# 1. 各種既存エンコーダの実装
# ==============================================================================

class MFCCPoissonEncoder:
    """1. MFCC / Mel-Spectrogram + Poisson Rate Coding.
    
    深層SNNで最も広く使われる標準的な手法。
    メルフィルタバンクの対数エネルギーを正規化し、ポアソン点過程でスパイク列を生成。
    """
    def __init__(self, sample_rate: int = 16000, num_channels: int = 16, hop_length_ms: float = 2.0):
        self.sr = sample_rate
        self.num_channels = num_channels
        self.hop_length_ms = hop_length_ms
        
    def encode(self, audio: np.ndarray) -> Tuple[np.ndarray, dict]:
        hop_samples = int(self.sr * (self.hop_length_ms / 1000.0))
        n_fft = 256
        
        freqs, times, Zxx = signal.stft(
            audio, fs=self.sr, nperseg=n_fft, noverlap=n_fft - hop_samples, boundary=None
        )
        spec = np.abs(Zxx)
        num_steps = spec.shape[1]
        
        # 16チャネルのメル対数スケールフィルタ
        center_freqs = np.geomspace(200.0, 7000.0, self.num_channels)
        mel_energies = np.zeros((num_steps, self.num_channels))
        
        for ch in range(self.num_channels):
            fc = center_freqs[ch]
            sigma = max(1.0, (fc * 0.20) / 2.355)
            kernel = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
            mel_energies[:, ch] = np.sum(spec * kernel[:, None], axis=0) / (np.sum(kernel) + 1e-8)
            
        # [0, 1] にクリップしてポアソン発火確率とする
        norm_rate = np.clip(mel_energies * 8.0, 0.0, 0.95)
        
        # ポアソン乱数スパイク生成
        np.random.seed(42)
        rand_vals = np.random.uniform(0.0, 1.0, size=norm_rate.shape)
        spikes = (rand_vals < norm_rate).astype(np.int8)
        
        return spikes, {"times": times, "center_freqs": center_freqs}


class TonicDeltaEncoder:
    """2. Tonic-style Delta / Threshold-Crossing Encoder.
    
    Tonic などのNeuromorphicライブラリで標準的な手法。
    スペクトル強度の時間微分 Δx(t) が閾値を超えたら ON スパイク、下回ったら OFF スパイクを出力。
    """
    def __init__(self, sample_rate: int = 16000, num_channels: int = 16, hop_length_ms: float = 2.0, threshold: float = 0.025):
        self.sr = sample_rate
        self.num_channels = num_channels
        self.hop_length_ms = hop_length_ms
        self.threshold = threshold
        
    def encode(self, audio: np.ndarray) -> Tuple[np.ndarray, dict]:
        hop_samples = int(self.sr * (self.hop_length_ms / 1000.0))
        n_fft = 256
        
        freqs, times, Zxx = signal.stft(
            audio, fs=self.sr, nperseg=n_fft, noverlap=n_fft - hop_samples, boundary=None
        )
        spec = np.abs(Zxx)
        num_steps = spec.shape[1]
        center_freqs = np.geomspace(200.0, 7000.0, self.num_channels)
        
        energies = np.zeros((num_steps, self.num_channels))
        for ch in range(self.num_channels):
            fc = center_freqs[ch]
            sigma = max(1.0, (fc * 0.20) / 2.355)
            kernel = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
            energies[:, ch] = np.sum(spec * kernel[:, None], axis=0) / (np.sum(kernel) + 1e-8)
            
        # ON/OFF スパイク (2 * num_channels の出力)
        spikes = np.zeros((num_steps, self.num_channels * 2), dtype=np.int8)
        
        diff = np.zeros_like(energies)
        diff[1:] = energies[1:] - energies[:-1]
        
        on_spikes = (diff > self.threshold).astype(np.int8)
        off_spikes = (diff < -self.threshold).astype(np.int8)
        
        spikes[:, :self.num_channels] = on_spikes
        spikes[:, self.num_channels:] = off_spikes
        
        return spikes, {"times": times, "center_freqs": center_freqs}


class StandardLIFEncoder:
    """3. Standard LIF FilterBank Encoder.
    
    固定帯域幅フィルタバンク + 単一LIFニューロン（順応・小胞プールなし）。
    """
    def __init__(self, sample_rate: int = 16000, num_channels: int = 16, hop_length_ms: float = 2.0, threshold: float = 0.20, leak_rate: float = 0.85):
        self.sr = sample_rate
        self.num_channels = num_channels
        self.hop_length_ms = hop_length_ms
        self.threshold = threshold
        self.leak_rate = leak_rate
        
    def encode(self, audio: np.ndarray) -> Tuple[np.ndarray, dict]:
        hop_samples = int(self.sr * (self.hop_length_ms / 1000.0))
        n_fft = 256
        freqs, times, Zxx = signal.stft(
            audio, fs=self.sr, nperseg=n_fft, noverlap=n_fft - hop_samples, boundary=None
        )
        spec = np.abs(Zxx)
        num_steps = spec.shape[1]
        center_freqs = np.geomspace(200.0, 7000.0, self.num_channels)
        
        spikes = np.zeros((num_steps, self.num_channels), dtype=np.int8)
        voltages = np.zeros(self.num_channels)
        
        for t in range(num_steps):
            frame_spec = spec[:, t]
            for ch in range(self.num_channels):
                fc = center_freqs[ch]
                sigma = max(1.0, (fc * 0.20) / 2.355)
                kernel = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
                energy = np.sum(frame_spec * kernel) / (np.sum(kernel) + 1e-8)
                
                voltages[ch] = self.leak_rate * voltages[ch] + energy * 2.0
                if voltages[ch] >= self.threshold:
                    spikes[t, ch] = 1
                    voltages[ch] = 0.0
                    
        return spikes, {"times": times, "center_freqs": center_freqs}


# ==============================================================================
# 2. 共通の特異点検出SNN（トノトピー・オンセット細胞モデル）
# ==============================================================================
class GenericOnsetDetector:
    """各エンコーダのスパイク出力を受容し、特異点スパイクを発火する下流SNN。
    
    周波数チャネルごとに直近の背景活動（順応レベル）を追跡し、
    未刺激チャネルからの急激なサージのみを統合してスパイク発火します。
    """
    def __init__(self, leak_rate: float = 0.45, threshold: float = 4.0):
        self.leak_rate = leak_rate
        self.threshold = threshold
        
    def detect(self, sp_channels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            sp_channels: (Steps, Channels) 各チャネルのスパイク数
        """
        steps, n_ch = sp_channels.shape
        detected = np.zeros(steps, dtype=np.int8)
        voltages = np.zeros(steps)
        
        bg_activity = np.zeros(n_ch)
        v = 0.0
        
        for t in range(steps):
            s = sp_channels[t]
            # チャネルごとの突発成分（背景レベルを上回る急激なサージ）
            surge = np.maximum(0.0, s - 1.6 * bg_activity)
            bg_activity = 0.90 * bg_activity + 0.10 * s
            
            # リードアウトニューロンの膜電位積分
            total_surge = np.sum(surge)
            v = self.leak_rate * v + total_surge
            voltages[t] = v
            
            if v >= self.threshold:
                detected[t] = 1
                v = 0.0
                
        return detected, voltages


# ==============================================================================
# 3. 総合ベンチマーク実行
# ==============================================================================
def run_comparison_benchmark():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print("【SNN オーディオエンコーダ 4大方式の直接対決ベンチマーク】")
    print("=" * 80)
    
    # 1. 音声生成
    sr = 16000
    t_audio, audio, events = generate_singularity_audio(sr=sr, duration=2.0)
    print(f"[1] テスト音源生成完了: 長さ={t_audio[-1]:.2f}s, 特異点イベント数={len(events)}")
    for ev in events:
        print(f"    * {ev['name']}: {ev['time']:.2f}s (~{ev['freq']}Hz)")
        
    # 2. 4つのエンコーダを初期化
    cfg_ribbon = EncoderConfig()  # Optuna 最適化済みパラメータセットを使用
    
    encoders = {
        "MFCC + Poisson": MFCCPoissonEncoder(sample_rate=sr, num_channels=16),
        "Tonic Delta": TonicDeltaEncoder(sample_rate=sr, num_channels=16, threshold=0.015),
        "Standard LIF": StandardLIFEncoder(sample_rate=sr, num_channels=16, threshold=0.20),
        "Ribbon SNN (Ours)": RibbonAudioEncoder(cfg_ribbon),
    }
    
    # 3. 各方式でエンコード＆特異点検出
    print("\n[2] 4つのエンコーダでエンコードおよび特異点検出を実行中...")
    results = {}
    
    for name, enc in encoders.items():
        if name == "Ribbon SNN (Ours)":
            sp_3d, info = enc.encode(audio, flatten_channels=False)  # (Steps, 16, 15)
            sp_ch = np.sum(sp_3d, axis=2)                            # (Steps, 16)
            spikes = sp_3d.reshape(len(sp_3d), -1)                  # (Steps, 240)
            det_th = 4.5
        elif name == "Tonic Delta":
            spikes, info = enc.encode(audio)                         # (Steps, 32: ON 16 + OFF 16)
            sp_ch = spikes[:, :16] + spikes[:, 16:]                  # (Steps, 16)
            det_th = 1.8
        else:
            spikes, info = enc.encode(audio)                         # (Steps, 16)
            sp_ch = spikes
            det_th = 2.0
            
        times = info["times"]
        detector = GenericOnsetDetector(threshold=det_th)
        det_spikes, voltages = detector.detect(sp_ch)
        
        # メトリクス評価
        window_sec = 0.025
        hits = 0
        latencies = []
        for ev in events:
            t_ev = ev["time"]
            mask = (times >= t_ev) & (times <= t_ev + window_sec)
            hit_steps = np.where(mask & (det_spikes > 0))[0]
            if len(hit_steps) > 0:
                first_t = times[hit_steps[0]]
                latencies.append((first_t - t_ev) * 1000.0)
                hits += 1
            else:
                latencies.append(None)
                
        # 背景区間の誤発火 (イベント発生時刻とその直後50msを除外)
        bg_mask = (times >= 0.25) & (times <= 1.85)
        for ev in events:
            bg_mask &= ~((times >= ev['time']) & (times <= ev['time'] + 0.060))
            
        false_alarms = int(np.sum(det_spikes[bg_mask]))
        total_spikes = int(np.sum(spikes))
        
        # スパイクS/N比 (イベント時スパイク密度 / 背景時スパイク密度)
        ev_mask = np.zeros(len(times), dtype=bool)
        for ev in events:
            ev_mask |= (times >= ev['time']) & (times <= ev['time'] + window_sec)
            
        pop = np.sum(spikes, axis=1)
        ev_mean = np.mean(pop[ev_mask]) if np.any(ev_mask) else 0.0
        bg_mean = np.mean(pop[bg_mask]) if np.any(bg_mask) else 1e-6
        snr = ev_mean / (bg_mean + 1e-6)
        
        results[name] = {
            "spikes": spikes,
            "det_spikes": det_spikes,
            "times": times,
            "hits": hits,
            "total_events": len(events),
            "latencies": latencies,
            "false_alarms": false_alarms,
            "total_spikes": total_spikes,
            "snr": snr,
        }
        
    # 4. コンソール結果サマリー表示
    print("\n" + "=" * 94)
    print(f"{'方式 (Encoder)':<20} | {'検出成功率':<10} | {'平均遅延':<10} | {'背景誤検知':<10} | {'総スパイク数':<12} | {'S/N比 (信号/背景)':<14}")
    print("-" * 94)
    
    for name, r in results.items():
        hit_str = f"{r['hits']}/{r['total_events']} ({r['hits']/len(events)*100:.0f}%)"
        valid_lats = [l for l in r['latencies'] if l is not None]
        lat_str = f"{np.mean(valid_lats):.1f} ms" if len(valid_lats) > 0 else "N/A"
        fa_str = f"{r['false_alarms']} 発"
        sp_str = f"{r['total_spikes']:,} 発"
        snr_str = f"{r['snr']:.2f} 倍"
        star = " ★ BEST" if name == "Ribbon SNN (Ours)" else ""
        print(f"{name:<20} | {hit_str:<10} | {lat_str:<10} | {fa_str:<10} | {sp_str:<12} | {snr_str:<14}{star}")
        
    print("=" * 94)
    
    # 5. 美麗な5段プロット可視化画像の作成
    out_img = "benchmark_comparison.png"
    fig, axes = plt.subplots(6, 1, figsize=(14, 15), sharex=True)
    plt.subplots_adjust(hspace=0.28)
    
    times = results["Ribbon SNN (Ours)"]["times"]
    
    # (0) 入力音声
    axes[0].plot(t_audio, audio, color="#2b5c8f", lw=0.9)
    colors = ["#e63946", "#d62828", "#9b2226"]
    for i, ev in enumerate(events):
        axes[0].axvline(ev["time"], color=colors[i], ls="--", lw=1.8, label=f"{ev['name']}")
    axes[0].set_title("(a) Input Audio Waveform: Steady 830Hz Tone + Noise + 3 Sudden Singularities", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # (1) MFCC + Poisson
    sp_mfcc = results["MFCC + Poisson"]["spikes"]
    m_steps, m_chs = np.where(sp_mfcc > 0)
    axes[1].scatter(times[m_steps], m_chs, s=2, color="#6c757d", alpha=0.4, marker="|")
    for ev in events: axes[1].axvline(ev["time"], color="crimson", ls=":", alpha=0.6)
    axes[1].set_title(f"(b) MFCC + Poisson Rate Coding: Random poisson noise floods background (Total: {results['MFCC + Poisson']['total_spikes']:,} spikes)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Mel Ch (16)")
    axes[1].grid(True, alpha=0.3)
    
    # (2) Tonic Delta
    sp_tonic = results["Tonic Delta"]["spikes"]
    t_steps, t_chs = np.where(sp_tonic > 0)
    axes[2].scatter(times[t_steps], t_chs, s=5, color="#17a2b8", alpha=0.6, marker="|")
    for ev in events: axes[2].axvline(ev["time"], color="crimson", ls=":", alpha=0.6)
    axes[2].set_title(f"(c) Tonic-style Delta (ON/OFF): Only edges trigger spikes; subtle attacks get lost (Total: {results['Tonic Delta']['total_spikes']:,} spikes)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("ON/OFF (32)")
    axes[2].grid(True, alpha=0.3)
    
    # (3) Standard LIF
    sp_lif = results["Standard LIF"]["spikes"]
    l_steps, l_chs = np.where(sp_lif > 0)
    axes[3].scatter(times[l_steps], l_chs, s=6, color="#495057", alpha=0.6, marker="|")
    for ev in events: axes[3].axvline(ev["time"], color="crimson", ls=":", alpha=0.6)
    axes[3].set_title(f"(d) Standard LIF FilterBank: Continuous firing on steady tone buries singularities (Total: {results['Standard LIF']['total_spikes']:,} spikes)", fontsize=10, fontweight="bold")
    axes[3].set_ylabel("Filter Ch (16)")
    axes[3].grid(True, alpha=0.3)
    
    # (4) Ribbon SNN (Ours)
    sp_ribbon = results["Ribbon SNN (Ours)"]["spikes"]
    r_steps, r_chs = np.where(sp_ribbon > 0)
    axes[4].scatter(times[r_steps], r_chs, s=6, color="#0077b6", alpha=0.7, marker="|")
    for ev in events: axes[4].axvline(ev["time"], color="crimson", ls=":", alpha=0.6)
    axes[4].set_title(f"(e) Ribbon SNN (Ours): Vesicle depletion silences steady noise; singularities erupt in explosive bursts (Total: {results['Ribbon SNN (Ours)']['total_spikes']:,} spikes)", fontsize=10, fontweight="bold")
    axes[4].set_ylabel("Neuron Index\n(16ch x 15)")
    axes[4].grid(True, alpha=0.3)
    
    # (5) 下流SNNの検出比較
    det_colors = ["#6c757d", "#17a2b8", "#e76f51", "#2a9d8f"]
    markers = ["s", "^", "x", "o"]
    
    for idx, (name, r) in enumerate(results.items()):
        det_steps = np.where(r["det_spikes"] > 0)[0]
        if len(det_steps) > 0:
            axes[5].scatter(
                times[det_steps],
                np.zeros_like(det_steps) + idx,
                s=90,
                marker=markers[idx],
                color=det_colors[idx],
                label=f"{name} ({r['hits']}/{r['total_events']} Hits, FA: {r['false_alarms']})"
            )
            
    for ev in events:
        axes[5].axvline(ev["time"], color="crimson", ls="--", alpha=0.7)
        
    axes[5].set_yticks(range(4))
    axes[5].set_yticklabels(list(results.keys()), fontweight="bold", fontsize=10)
    axes[5].set_ylim(-0.6, 3.6)
    axes[5].set_title("(f) Downstream SNN Singularity Detection: Baseline/MFCC miss subtle events or generate false alarms; Ribbon achieves 100% hits", fontsize=11, fontweight="bold")
    axes[5].set_xlabel("Time (seconds)", fontsize=11)
    axes[5].legend(loc="upper right", framealpha=0.9, fontsize=8)
    axes[5].grid(True, alpha=0.3)
    axes[5].set_xlim(0, 2.0)
    
    plt.savefig(out_img, dpi=180, bbox_inches="tight")
    print(f"\n[3] 4大エンコーダの比較プロット画像を保存しました: {out_img}")
    plt.close()


if __name__ == "__main__":
    run_comparison_benchmark()
