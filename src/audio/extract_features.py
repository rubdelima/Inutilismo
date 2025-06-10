import numpy as np
import librosa
import pyworld as pw #type:ignore
from pathlib import Path
import soundfile as sf #type:ignore
import shutil

from .separate import separate_vocals_advanced
from .preprocess import preprocess_audio_for_rvc
from src.core import get_logger

logger = get_logger('extract_features', console=False)

def extract_vocal_features(audio: np.ndarray, sr: int = 40000) -> dict:
    """
    Extrai características vocais para RVC
    
    Args:
        audio: Áudio de entrada
        sr: Sample rate
    
    Returns:
        dict: Características extraídas
    """
    logger.info("🎤 Extraindo características vocais...")
    
    # F0 (pitch) usando WORLD
    f0, timeaxis = pw.harvest(audio.astype(np.float64), sr)
    f0 = pw.stonemask(audio.astype(np.float64), f0, timeaxis, sr)
    
    # Spectral features
    sp = pw.cheaptrick(audio.astype(np.float64), f0, timeaxis, sr)
    ap = pw.d4c(audio.astype(np.float64), f0, timeaxis, sr)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Spectral centroid, rolloff, etc
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    features = {
        'f0': f0,
        'sp': sp,
        'ap': ap,
        'mfcc': mfcc,
        'spectral_centroids': spectral_centroids,
        'spectral_rolloff': spectral_rolloff,
        'timeaxis': timeaxis
    }
    
    logger.info("✅ Características extraídas")
    return features

def extract_vocal_features_fixed(audio: np.ndarray, sr: int = 40000, target_frames: int = 200) -> dict:
    """
    Extrai características vocais para RVC com tamanho fixo
    
    Args:
        audio: Áudio de entrada
        sr: Sample rate
        target_frames: Número alvo de frames para MFCC
    
    Returns:
        dict: Características extraídas com tamanhos normalizados
    """
    logger.info("🎤 Extraindo características vocais (tamanho fixo)...")
    
    # F0 (pitch) usando WORLD
    f0, timeaxis = pw.harvest(audio.astype(np.float64), sr)
    f0 = pw.stonemask(audio.astype(np.float64), f0, timeaxis, sr)
    
    # Spectral features
    sp = pw.cheaptrick(audio.astype(np.float64), f0, timeaxis, sr)
    ap = pw.d4c(audio.astype(np.float64), f0, timeaxis, sr)
    
    # MFCC com tamanho fixo
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Normalizar tamanho do MFCC
    current_frames = mfcc.shape[1]
    
    if current_frames > target_frames:
        # Truncar
        mfcc = mfcc[:, :target_frames]
    elif current_frames < target_frames:
        # Pad com zeros
        pad_width = target_frames - current_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    
    # Spectral centroid, rolloff, etc (também normalizar)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    # Normalizar spectral features
    if spectral_centroids.shape[1] > target_frames:
        spectral_centroids = spectral_centroids[:, :target_frames]
    elif spectral_centroids.shape[1] < target_frames:
        pad_width = target_frames - spectral_centroids.shape[1]
        spectral_centroids = np.pad(spectral_centroids, ((0, 0), (0, pad_width)), mode='edge')
    
    if spectral_rolloff.shape[1] > target_frames:
        spectral_rolloff = spectral_rolloff[:, :target_frames]
    elif spectral_rolloff.shape[1] < target_frames:
        pad_width = target_frames - spectral_rolloff.shape[1]
        spectral_rolloff = np.pad(spectral_rolloff, ((0, 0), (0, pad_width)), mode='edge')
    
    features = {
        'f0': f0,
        'sp': sp,
        'ap': ap,
        'mfcc': mfcc,
        'spectral_centroids': spectral_centroids,
        'spectral_rolloff': spectral_rolloff,
        'timeaxis': timeaxis
    }
    
    logger.info(f"✅ Características extraídas - MFCC: {mfcc.shape}")
    return features

def process_single_audio_file(args):
    """
    Processa um único arquivo de áudio (para paralelização)
    
    Args:
        args: tupla (audio_path, output_dir, target_sr, vocal_input, max_length)
    
    Returns:
        list: Lista de arquivos processados
    """
    audio_path, output_dir, target_sr, vocal_input, max_length, file_index = args
    
    try:
        logger.info(f"🔧 Processando arquivo {file_index}: {Path(audio_path).name}")
        
        if not vocal_input:
            # Separar vocal do instrumental
            temp_sep_dir = f"temp_separation_{file_index}"
            vocals_path, _ = separate_vocals_advanced(audio_path, temp_sep_dir)
            source_audio = vocals_path
        else:
            source_audio = audio_path
        
        # Pré-processar áudio
        audio_data = preprocess_audio_for_rvc(source_audio, target_sr)
        
        # Quebrar em segmentos
        if len(audio_data) > max_length:
            segments = [audio_data[i:i+max_length] 
                      for i in range(0, len(audio_data), max_length)]
        else:
            segments = [audio_data]
        
        # Salvar segmentos processados
        processed_files = []
        for j, segment in enumerate(segments):
            if len(segment) < target_sr:  # Ignorar segmentos muito curtos
                continue
                
            output_path = Path(output_dir) / f"audio_{file_index:03d}_{j:02d}.wav"
            sf.write(output_path, segment, target_sr)
            processed_files.append(str(output_path))
        
        # Limpar arquivos temporários
        if not vocal_input:
            shutil.rmtree(f"temp_separation_{file_index}", ignore_errors=True)
        
        logger.info(f"✅ Processado arquivo {file_index}: {len(processed_files)} segmentos")
        return processed_files
        
    except Exception as e:
        logger.error(f"❌ Erro ao processar arquivo {file_index}: {e}")
        return []

def process_single_audio_file_sequential(audio_path, output_dir, target_sr, vocal_input, max_length, file_index):
    """
    Processa um único arquivo de áudio SEQUENCIALMENTE (sem paralelização)
    
    Args:
        audio_path: caminho do arquivo
        output_dir: diretório de saída
        target_sr: sample rate alvo
        vocal_input: se já é vocal isolado
        max_length: tamanho máximo do segmento
        file_index: índice do arquivo
    
    Returns:
        list: Lista de arquivos processados
    """
    try:
        logger.info(f"🔧 Processando arquivo {file_index}: {Path(audio_path).name}")
        
        if not vocal_input:
            # Separar vocal do instrumental SEQUENCIALMENTE
            temp_sep_dir = f"temp_separation_{file_index}"
            vocals_path, _ = separate_vocals_advanced(audio_path, temp_sep_dir)
            source_audio = vocals_path
        else:
            source_audio = audio_path
        
        # Pré-processar áudio
        audio_data = preprocess_audio_for_rvc(source_audio, target_sr)
        
        # Quebrar em segmentos
        if len(audio_data) > max_length:
            segments = [audio_data[i:i+max_length] 
                      for i in range(0, len(audio_data), max_length)]
        else:
            segments = [audio_data]
        
        # Salvar segmentos processados
        processed_files = []
        for j, segment in enumerate(segments):
            if len(segment) < target_sr:  # Ignorar segmentos muito curtos
                continue
                
            output_path = Path(output_dir) / f"audio_{file_index:03d}_{j:02d}.wav"
            sf.write(output_path, segment, target_sr)
            processed_files.append(str(output_path))
        
        # Limpar arquivos temporários
        if not vocal_input:
            shutil.rmtree(f"temp_separation_{file_index}", ignore_errors=True)
        
        logger.info(f"✅ Processado arquivo {file_index}: {len(processed_files)} segmentos")
        return processed_files
        
    except Exception as e:
        logger.error(f"❌ Erro ao processar arquivo {file_index}: {e}")
        return []

def process_features_parallel(audio_file):
    """
    Processa features de um arquivo (ESTA função é paralelizada)
    
    Args:
        audio_file: caminho do arquivo
    
    Returns:
        tuple: (features_flattened, feature_path)
    """
    try:
        target_sr = 40000
        target_frames = 200  # Tamanho fixo para normalização
        
        logger.info(f"🔍 Extraindo features: {audio_file.name}")
        
        audio, _ = librosa.load(audio_file, sr=target_sr, mono=True)
        features = extract_vocal_features_fixed(audio, target_sr, target_frames)
        
        # Salvar características individuais
        feature_path = audio_file.parent.parent / f"features_{audio_file.stem}.npz"
        np.savez_compressed(feature_path, **features)
        
        # Retornar MFCC flattened para índice global
        mfcc_flat = features['mfcc'].flatten()
        
        logger.info(f"✅ Features extraídas: {audio_file.name}")
        return mfcc_flat, str(feature_path)
        
    except Exception as e:
        logger.error(f"❌ Erro ao extrair features de {audio_file}: {e}")
        return None, None