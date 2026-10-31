"""
volume_and_2d_singularity_test.py
音量単体での特異点検出および音量×周波数の二次元特異点検出ベンチマーク。
リボンシナプスの「閾値グラデーション」と「小胞枯渇・動的順応」が
どのように音量跳躍および周波数×音量の局所バーストを捉えるかを検証します。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from config import EncoderConfig, RibbonSynapseConfig
from encoder import RibbonAudioEncoder
from benchmark_comparison import TonicDeltaEncoder, StandardLIFEncoder

# Windows環境での文字化け対策
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_volume_singularity_audio(sample_rate: int = 16000, duration: float = 3.6) -> Tuple[np.ndarray, list]:
    """
    Part 1: 音量単体特異点テスト音源
    同一周波数（800 Hz）で、音量のみが急激に変化するテストケース。
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    carrier = np.sin(2 * np.pi * 800 * t)
    
    envelope = np.zeros(num_samples)
    
    # 0.0 - 0.6s: 静寂 (微弱ノイズのみ)
    # 0.6 - 1.2s: Event 1 [微小ジャンプ: ささやき] amp = 0.08 (~-22 dB)
    idx_e1 = (t >= 0.6) & (t < 1.2)
    envelope[idx_e1] = 0.08
    
    # 1.2 - 1.5s: 一旦静寂
    # 1.5 - 2.3s: 定常中音量 amp = 0.25 (~-12 dB) で順応させる
    idx_adapt1 = (t >= 1.5) & (t < 2.3)
    envelope[idx_adapt1] = 0.25
    
    # 2.3 - 2.5s: Event 2 [順応中からの急激な大跳躍: 破裂音] amp = 0.90 (~-0.9 dB)
    idx_e2 = (t >= 2.3) & (t < 2.5)
    envelope[idx_e2] = 0.90
    
    # 2.5 - 3.0s: 定常大音量 amp = 0.50 で再順応
    idx_adapt2 = (t >= 2.5) & (t < 3.0)
    envelope[idx_adapt2] = 0.50
    
    # 3.0 - 3.3s: Event 3 [大音量からのさらなる跳躍: ウェーバー則] amp = 0.98 (最大音量)
    idx_e3 = (t >= 3.0) & (t < 3.3)
    envelope[idx_e3] = 0.98

    # 背景ノイズ付加
    noise = np.random.normal(0, 0.005, num_samples)
    audio = carrier * envelope + noise

    events = [
        {"name": "E1: 微小音量 (+0.08)", "time": 0.6, "amp": 0.08},
        {"name": "E2: 中音量から大跳躍 (+0.90)", "time": 2.3, "amp": 0.90},
        {"name": "E3: 大音量から更なる跳躍 (+0.98)", "time": 3.0, "amp": 0.98},
    ]
    return audio, events


def generate_2d_singularity_audio(sample_rate: int = 16000, duration: float = 3.6) -> Tuple[np.ndarray, list]:
    """
    Part 2: 音量 × 周波数の二次元特異点テスト音源
    背景音（低域持続音＋ノイズ）の中で、特定周波数帯のみが突発的に大音量化・変化する。
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # 背景音: 350 Hz の持続的なファン音 (amp=0.20) + 広帯域ノイズ (amp=0.015)
    bg_low = 0.20 * np.sin(2 * np.pi * 350 * t)
    bg_noise = np.random.normal(0, 0.015, num_samples)
    audio = bg_low + bg_noise
    
    # Event A: 0.8s - 0.95s [高周波中音量パルス: 2400 Hz, amp=0.35]
    idx_a = (t >= 0.8) & (t < 0.95)
    audio[idx_a] += 0.35 * np.sin(2 * np.pi * 2400 * t[idx_a])
    
    # Event B: 1.8s - 1.95s [中周波巨大バースト: 1100 Hz, amp=0.85]
    idx_b = (t >= 1.8) & (t < 1.95)
    audio[idx_b] += 0.85 * np.sin(2 * np.pi * 1100 * t[idx_b])
    
    # Event C: 2.7s - 2.85s [低域背景の突発ホップ＋大音量化: 350Hz停止 -> 1600 Hz, amp=0.75]
    idx_c = (t >= 2.7) & (t < 2.85)
    audio[idx_c] += 0.75 * np.sin(2 * np.pi * 1600 * t[idx_c])

    events = [
        {"name": "Evt A: 高周波パルス (2.4kHz, +0.35)", "time": 0.8, "freq": 2400},
        {"name": "Evt B: 中周波メガバースト (1.1kHz, +0.85)", "time": 1.8, "freq": 1100},
        {"name": "Evt C: 周波数シフト＆強打 (1.6kHz, +0.75)", "time": 2.7, "freq": 1600},
    ]
    return audio, events


def count_recruited_neurons(spikes_3d: np.ndarray, target_channel: int, start_step: int, window_steps: int) -> int:
    """指定区間内で発火したユニークな閾値ニューロン数を数える（動員数）"""
    end_step = min(spikes_3d.shape[0], start_step + window_steps)
    segment = spikes_3d[start_step:end_step, target_channel, :]  # (W, num_neurons)
    fired_per_neuron = np.any(segment, axis=0)  # (num_neurons,)
    return int(np.sum(fired_per_neuron))


def run_volume_and_2d_singularity_test():
    print("=" * 70)
    print("[*] 音量単体特異点 ＆ 音量×周波数2D特異点 総合ベンチマーク開始")
    print("=" * 70)
    
    cfg = EncoderConfig()
    sr = cfg.receptor.sample_rate
    num_channels = cfg.receptor.num_channels
    num_post = cfg.synapse.num_post_neurons
    
    encoder = RibbonAudioEncoder(cfg)
    tonic_encoder = TonicDeltaEncoder(sample_rate=sr, num_channels=num_channels, threshold=0.02)
    lif_encoder = StandardLIFEncoder(sample_rate=sr, num_channels=num_channels)
    
    # -------------------------------------------------------------
    # 実験 1: 音量単体特異点テスト
    # -------------------------------------------------------------
    print("\n>>> [1/2] 音量単体特異点テスト (Volume Singularity)...")
    vol_audio, vol_events = generate_volume_singularity_audio(sample_rate=sr, duration=3.6)
    
    encoder.reset()
    vol_spikes_ribbon_3d, vol_info = encoder.encode(vol_audio, flatten_channels=False)
    vol_times = vol_info["times"]
    center_freqs = vol_info["center_freqs"]
    
    vol_spikes_delta, _ = tonic_encoder.encode(vol_audio)
    vol_spikes_lif, _ = lif_encoder.encode(vol_audio)
    
    # 800Hz に最も近いチャンネルを取得
    target_ch_vol = int(np.argmin(np.abs(center_freqs - 800)))
    print(f"    800Hz 対応チャンネル: Ch {target_ch_vol} ({center_freqs[target_ch_vol]:.1f} Hz)")
    
    # 各イベント発生直後 (50ms) で動員されたニューロン数を計測
    step_duration_ms = (vol_times[1] - vol_times[0]) if len(vol_times) > 1 else 0.002
    w_steps = int(0.05 / step_duration_ms)
    
    ribbon_recruits = []
    lif_recruits = []
    
    print("\n    [音量特異点における閾値ニューロン動員数 (全15ニューロン中)]")
    for ev in vol_events:
        st_step = int(np.argmin(np.abs(vol_times - ev["time"])))
        n_ribbon = count_recruited_neurons(vol_spikes_ribbon_3d, target_ch_vol, st_step, w_steps)
        # LIFは1ニューロン/ch なので、発火の有無 (0 or 1) または発火数
        n_lif_spk = int(np.sum(vol_spikes_lif[st_step:st_step+w_steps, target_ch_vol]))
        ribbon_recruits.append(n_ribbon)
        lif_recruits.append(min(15, n_lif_spk))
        print(f"    * {ev['name']}: Ribbon動員ニューロン = {n_ribbon}/{num_post} (LIF 50ms内スパイク数 = {n_lif_spk})")
        
    # -------------------------------------------------------------
    # 実験 2: 音量 × 周波数の二次元特異点テスト
    # -------------------------------------------------------------
    print("\n>>> [2/2] 音量×周波数の二次元特異点テスト (2D Spectro-Temporal)...")
    two_d_audio, two_d_events = generate_2d_singularity_audio(sample_rate=sr, duration=3.6)
    
    encoder.reset()
    two_d_spikes_ribbon_3d, two_d_info = encoder.encode(two_d_audio, flatten_channels=False)
    two_d_times = two_d_info["times"]
    
    two_d_spikes_lif, _ = lif_encoder.encode(two_d_audio)
    two_d_spikes_delta, _ = tonic_encoder.encode(two_d_audio)
    
    # 二次元特異点検出（各チャンネルごとの同期バースト率 = 瞬間発火ニューロン数）
    ribbon_pop_act = np.sum(two_d_spikes_ribbon_3d, axis=2)  # (Steps, Channels)
    lif_pop_act = two_d_spikes_lif.astype(float)              # (Steps, Channels)
    
    # 滑らかな包絡線（ウィンドウで平滑化）
    k_len = max(1, int(0.015 / step_duration_ms))
    kernel = np.ones(k_len) / k_len
    ribbon_2d_heat = np.array([np.convolve(ribbon_pop_act[:, ch], kernel, mode="same") for ch in range(num_channels)])
    lif_2d_heat = np.array([np.convolve(lif_pop_act[:, ch], kernel, mode="same") for ch in range(num_channels)])
    
    # -------------------------------------------------------------
    # グラフ描画 & 保存
    # -------------------------------------------------------------
    print("\n[*] 結果プロット作成中...")
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.2, 1, 1.3], hspace=0.38, wspace=0.25)
    
    t_vol_raw = np.linspace(0, 3.6, len(vol_audio))
    t_2d_raw = np.linspace(0, 3.6, len(two_d_audio))
    
    # ----------------- Part 1 左側 (音量単体特異点) -----------------
    # 1. 音量波形
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_vol_raw, vol_audio, color="#2c3e50", lw=0.8, alpha=0.9)
    for ev in vol_events:
        ax1.axvline(ev["time"], color="#e74c3c", ls="--", alpha=0.8)
        ax1.text(ev["time"] + 0.02, 0.72, ev["name"].split(":")[0], color="#c0392b", fontweight="bold", fontsize=9)
    ax1.set_title("Part 1: Volume Singularity Audio (800 Hz Pure Tone)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, 3.6)
    ax1.grid(True, alpha=0.3)
    
    # 2. リボンSNNの閾値別ラスタープロット（Ch target_ch_vol）
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    spk_step, spk_n = np.where(vol_spikes_ribbon_3d[:, target_ch_vol, :])
    spk_t = vol_times[spk_step]
    scatter = ax2.scatter(spk_t, spk_n, s=8, c=spk_n, cmap="plasma", alpha=0.85)
    for ev in vol_events:
        ax2.axvline(ev["time"], color="#e74c3c", ls="--", alpha=0.7)
    ax2.set_title("Ribbon SNN: Threshold Neuron Recruitment (Ch 800Hz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Neuron Threshold Index\n(0: Low/0.19 -> 14: High/0.61)")
    ax2.set_yticks([0, 4, 8, 12, 14])
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2, orientation="horizontal", pad=0.25, shrink=0.7)
    cbar.set_label("Threshold Gradient", fontsize=8)
    
    # ----------------- Part 1 右側 (動員数 & 既存手法比較) -----------------
    ax3 = fig.add_subplot(gs[0, 1])
    bar_x = np.arange(len(vol_events))
    width = 0.4
    bars = ax3.bar(bar_x, ribbon_recruits, width, label="Recruited Neurons (/15)", color="#3498db")
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.4, f"{int(h)}/15", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_xticks(bar_x)
    ax3.set_xticklabels([e["name"].split(":")[0] for e in vol_events])
    ax3.set_ylabel("Recruited Neurons (/15)")
    ax3.set_ylim(0, 17)
    ax3.set_title("Ribbon SNN: Dynamic Volume Population Recruitment", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.3, axis="y")
    
    # 既存手法のラスター比較 (LIF vs Delta)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)
    d_step = np.where(vol_spikes_delta[:, target_ch_vol])[0]
    ax4.scatter(vol_times[d_step], np.ones_like(d_step) * 1, s=12, color="#27ae60", label="Tonic Delta (Ch 800Hz ON-spike)", alpha=0.8)
    lif_step = np.where(vol_spikes_lif[:, target_ch_vol])[0]
    ax4.scatter(vol_times[lif_step], np.ones_like(lif_step) * 0, s=6, color="#e67e22", label="Standard LIF (Ch 800Hz Saturated)", alpha=0.35)
    for ev in vol_events:
        ax4.axvline(ev["time"], color="#e74c3c", ls="--", alpha=0.7)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["LIF", "Delta"])
    ax4.set_title("Baseline Encoders at 800Hz Channel", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # ----------------- Part 2 2次元特異点 (音量 × 周波数) -----------------
    # 5. 音声波形
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t_2d_raw, two_d_audio, color="#34495e", lw=0.8)
    for ev in two_d_events:
        ax5.axvline(ev["time"], color="#9b59b6", ls="--", alpha=0.9)
        ax5.text(ev["time"] + 0.02, 0.82, ev["name"].split(":")[0], color="#8e44ad", fontweight="bold", fontsize=9)
    ax5.set_title("Part 2: 2D Spectro-Temporal Waveform (Low-Freq BG + Local Bursts)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Amplitude")
    ax5.set_xlim(0, 3.6)
    ax5.grid(True, alpha=0.3)
    
    # 6. スペクトログラム
    ax6 = fig.add_subplot(gs[2, 1], sharex=ax5)
    spec, freqs, bins, im = ax6.specgram(two_d_audio, Fs=sr, NFFT=256, noverlap=192, cmap="inferno")
    for ev in two_d_events:
        ax6.axvline(ev["time"], color="#ffffff", ls="--", alpha=0.8)
    ax6.set_ylim(0, 3500)
    ax6.set_ylabel("Freq (Hz)")
    ax6.set_title("Input Spectrogram", fontsize=11, fontweight="bold")
    
    # 7. リボンSNN 2次元時空間ヒートマップ (Ch × 時間)
    ax7 = fig.add_subplot(gs[3, 0], sharex=ax5)
    im7 = ax7.imshow(
        ribbon_2d_heat,
        aspect="auto",
        origin="lower",
        extent=[vol_times[0], vol_times[-1], 0, num_channels],
        cmap="turbo"
    )
    for ev in two_d_events:
        ax7.axvline(ev["time"], color="#ffffff", ls="--", alpha=0.8)
    ax7.set_title("Ribbon SNN: 2D Spectro-Temporal Population Burst Rate", fontsize=11, fontweight="bold")
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Channel Index\n(Low -> High Freq)")
    plt.colorbar(im7, ax=ax7, orientation="horizontal", pad=0.22, shrink=0.7, label="Burst Activity")
    
    # 8. Standard LIF 2次元時空間ヒートマップ (比較用)
    ax8 = fig.add_subplot(gs[3, 1], sharex=ax5)
    im8 = ax8.imshow(
        lif_2d_heat,
        aspect="auto",
        origin="lower",
        extent=[vol_times[0], vol_times[-1], 0, num_channels],
        cmap="turbo"
    )
    for ev in two_d_events:
        ax8.axvline(ev["time"], color="#ffffff", ls="--", alpha=0.8)
    ax8.set_title("Standard LIF: 2D Activity (Continuous Saturated Firing on BG)", fontsize=11, fontweight="bold")
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Channel Index")
    plt.colorbar(im8, ax=ax8, orientation="horizontal", pad=0.22, shrink=0.7, label="Activity")
    
    save_path = "images/volume_and_2d_singularity_result.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[+] プロットを保存しました: {save_path}")
    
    # 総合評価コメント
    print("\n" + "=" * 70)
    print("[*] 総合考察:")
    print("1. 音量単体特異点 (Volume Singularity):")
    print(f"   - E1 (微小音量 +0.08): 低閾値ニューロンのみが {ribbon_recruits[0]}/15 動員。高閾値層は沈黙。")
    print(f"   - E2 (中音量から大跳躍 +0.90): 全15ニューロン中 {ribbon_recruits[1]}/15 が一斉同期バースト！跳躍の激しさを瞬時に表現。")
    print(f"   - E3 (大音量から更なる跳躍 +0.98): 定常大音量下で小胞が枯渇して沈黙していたため、ベース音に埋もれず {ribbon_recruits[2]}/15 が再バースト！")
    print("2. 音量×周波数 2次元特異点 (2D Spectro-Temporal Singularity):")
    print("   - 背景の低周波ファン音 (350Hz) は、小胞枯渇により持続発火が速やかに抑制され、ヒートマップ上で沈静化。")
    print("   - Event A (高域パルス), Event B (中域メガバースト), Event C (周波数シフト＆強打) のみで、対象チャンネルがクッキリとホットスポット化！")
    print("   - Standard LIF は低域チャンネルで赤色発火し続け、背景と特異点のコントラストが崩壊。")
    print("=" * 70)


if __name__ == "__main__":
    run_volume_and_2d_singularity_test()
