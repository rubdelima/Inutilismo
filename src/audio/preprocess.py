import librosa
import numpy as np

from src.core import get_logger

logger = get_logger("preprocess", console=False)

def preprocess_audio_for_rvc(audio_path: str, target_sr: int = 40000) -> np.ndarray:
    """
    Pré-processa áudio para RVC
    
    Args:
        audio_path: Caminho do áudio
        target_sr: Sample rate alvo
    
    Returns:
        np.ndarray: Áudio processado
    """
    logger.info(f"🔧 Pré-processando áudio: {audio_path}")
    
    # Carregar áudio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Resample se necessário
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        logger.warning(f"🔄 Resampled: {sr}Hz -> {target_sr}Hz")
    
    # Normalização
    audio = librosa.util.normalize(audio)
    
    # Filtro passa-alta para remover ruído baixo
    from scipy.signal import butter, filtfilt
    nyquist = target_sr // 2
    high_cutoff = 80 / nyquist  # 80Hz
    b, a = butter(5, high_cutoff, btype='high')
    audio = filtfilt(b, a, audio)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    logger.info(f"✅ Áudio processado: {len(audio)/target_sr:.1f}s")
    return audio