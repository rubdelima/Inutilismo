import os
import subprocess
from pathlib import Path
from src.core import get_logger

logger = get_logger("separate_audio", console=False)

def separate_vocals_advanced(audio_path: str, output_dir: str = "separated", method: str = "demucs") -> tuple:
    """
    Separa áudio em vocal e instrumental usando diferentes métodos
    
    Args:
        audio_path: Caminho do áudio
        output_dir: Diretório de saída
        method: Método de separação ('demucs', 'spleeter')
    
    Returns:
        tuple: (vocals_path, instrumental_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
    
    audio_name = Path(audio_path).stem
    
    if method == "demucs":
        return _separate_with_demucs(audio_path, output_dir, audio_name)
    elif method == "spleeter":
        return _separate_with_spleeter(audio_path, output_dir, audio_name)
    else:
        raise ValueError(f"Método não suportado: {method}")

def _separate_with_demucs(audio_path: str, output_dir: str, audio_name: str) -> tuple:
    """Separação usando Demucs"""
    logger.info("🎵 Separando com Demucs...")
    
    try:
        result = subprocess.run([
            "python", "-m", "demucs.separate", 
            "--two-stems=vocals", 
            "-o", output_dir, 
            audio_path
        ], capture_output=True, text=True, check=True)
        
        # Caminhos esperados
        base_dir = os.path.join(output_dir, "htdemucs", audio_name)
        vocals_path = os.path.join(base_dir, "vocals.wav")
        instrumental_path = os.path.join(base_dir, "no_vocals.wav")
        
        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Vocal não gerado: {vocals_path}")
            
        logger.info(f"✅ Separação concluída: {vocals_path}")
        return vocals_path, instrumental_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro no Demucs: {e}")
        raise

def _separate_with_spleeter(audio_path: str, output_dir: str, audio_name: str) -> tuple:
    """Separação usando Spleeter (fallback)"""
    logger.info("🎵 Separando com Spleeter...")
    
    try:
        result = subprocess.run([
            "spleeter", "separate", 
            "-p", "spleeter:2stems-16kHz", 
            "-o", output_dir, 
            audio_path
        ], capture_output=True, text=True, check=True)
        
        # Caminhos esperados do Spleeter
        base_dir = os.path.join(output_dir, audio_name)
        vocals_path = os.path.join(base_dir, "vocals.wav")
        instrumental_path = os.path.join(base_dir, "accompaniment.wav")
        
        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Vocal não gerado: {vocals_path}")
            
        logger.info(f"✅ Separação concluída: {vocals_path}")
        return vocals_path, instrumental_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro no Spleeter: {e}")
        raise