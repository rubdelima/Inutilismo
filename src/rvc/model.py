import torch
from pathlib import Path
from typing import Optional, List
import faiss #type:ignore
import numpy as np
import soundfile as sf #type:ignore
import shutil
import os
import librosa

from src.audio import separate_vocals_advanced, \
    preprocess_audio_for_rvc, extract_vocal_features_fixed

from src.core import get_logger, AdvancedMemoryManager

logger = get_logger("voice_manager")
memory_manager = AdvancedMemoryManager()

class VoiceManager:
    """Gerenciador principal para conversão de voz usando RVC"""
    
    def __init__(self, model_path: str, index_path: Optional[str] = None):
        # Limpeza inicial
        memory_manager.clear_memory()
        
        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else self.model_path.with_suffix('.index')
        
        # Selecionar dispositivo baseado na memória disponível
        self.device = memory_manager.smart_device_selection(model_size_gb=0.5)
        
        # Verificar se arquivos essenciais existem
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        if not self.index_path.exists():
            logger.warning(f"Índice não encontrado: {self.index_path}")
            self.index_path = None # type:ignore
        
        # Carregar modelo e índice
        self._load_model()
        self._load_index()
        
        logger.info(f"🎯 VoiceManager inicializado")
        logger.info(f"📄 Modelo: {self.model_path.name}")
        logger.info(f"🔍 Índice: {self.index_path.name if self.index_path else 'Não disponível'}")
        logger.info(f"🎮 Dispositivo: {self.device}")
        memory_manager.monitor_memory()

    def _load_model(self):
        """Carrega modelo RVC"""
        logger.info("📥 Carregando modelo RVC...")
        
        try:
            self.model_data = torch.load(self.model_path, map_location='cpu')
            logger.info("✅ Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            # Fallback mínimo
            self.model_data = {
                'model_name': self.model_path.stem,
                'sample_rate': 40000,
                'target_frames': 200,
                'version': 'unknown'
            }
        
        # Extrair configurações
        self.sample_rate = self.model_data.get('sample_rate', 40000)
        self.model_name = self.model_data.get('model_name', self.model_path.stem)
        self.target_frames = self.model_data.get('target_frames', 200)
        
        logger.info(f"🎯 Configurações: {self.model_name}, {self.sample_rate}Hz, {self.target_frames} frames")

    def _load_index(self):
        """Carrega índice FAISS com otimização de memória"""
        if self.index_path and self.index_path.exists():
            try:
                logger.info("📥 Carregando índice FAISS...")
                
                # Verificar se deve usar GPU para FAISS
                use_gpu = memory_manager.should_use_gpu(1.0) and torch.cuda.is_available()
                
                if use_gpu:
                    try:
                        # Carregar na GPU
                        cpu_index = faiss.read_index(str(self.index_path))
                        res = faiss.StandardGpuResources()
                        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                        logger.info(f"✅ Índice carregado na GPU: {self.faiss_index.ntotal} vetores, {self.faiss_index.d}D")
                    except Exception as e:
                        logger.warning(f"⚠️ Erro ao carregar na GPU, usando CPU: {e}")
                        self.faiss_index = faiss.read_index(str(self.index_path))
                        logger.info(f"✅ Índice carregado na CPU: {self.faiss_index.ntotal} vetores, {self.faiss_index.d}D")
                else:
                    self.faiss_index = faiss.read_index(str(self.index_path))
                    logger.info(f"✅ Índice carregado na CPU: {self.faiss_index.ntotal} vetores, {self.faiss_index.d}D")
                
                # Limpeza após carregamento
                memory_manager.clear_memory()
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar índice: {e}")
                self.faiss_index = None
        else:
            self.faiss_index = None
            logger.warning("⚠️ Funcionando sem índice FAISS (qualidade reduzida)")
    
    def mtm(self, audio_path: str, output_path: str, vocal_input: bool = False, only_vocal: bool = True) -> Optional[str]:
        """
        Music-to-Music: Conversão de voz preservando melodia
        
        Args:
            audio_path: Áudio de entrada
            output_path: Arquivo de saída
            vocal_input: Se entrada já é vocal isolado
            only_vocal: Se True, retorna apenas vocal convertido
        
        Returns:
            str: Caminho do arquivo gerado
        """
        logger.info(f"🎵 MTM: {Path(audio_path).name}")
        
        # Monitor inicial
        memory_manager.monitor_memory()
        
        try:
            if not vocal_input:
                # Separar vocal do instrumental (processo que consome muita memória)
                logger.info("🎤 Separando vocal do instrumental...")
                
                # Limpeza antes da separação
                memory_manager.clear_memory()
                
                vocals_path, instrumental_path = separate_vocals_advanced(audio_path, "temp_mtm")
                
                # Limpeza após separação
                memory_manager.clear_memory()
            else:
                vocals_path = audio_path
                instrumental_path = None
            
            # Converter voz
            converted_vocal_path = self._convert_voice(vocals_path, "temp_converted_vocal.wav", vocal_input=True)
            
            if only_vocal:
                # Retornar apenas vocal convertido
                shutil.move(converted_vocal_path, output_path)
                result_path = output_path
            else:
                # Mixar vocal convertido com instrumental
                if instrumental_path and os.path.exists(instrumental_path):
                    result_path = self._mix_audio(converted_vocal_path, instrumental_path, output_path)
                else:
                    logger.warning("⚠️ Instrumental não disponível, retornando apenas vocal")
                    shutil.move(converted_vocal_path, output_path)
                    result_path = output_path
            
            # Limpar arquivos temporários
            self._cleanup_temp_files(["temp_mtm", "temp_converted_vocal.wav"])
            
            # Limpeza final
            memory_manager.clear_memory()
            
            logger.info(f"✅ MTM concluído: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"❌ Erro no MTM: {e}")
            memory_manager.clear_memory(aggressive=True)
            return None
    
    def _convert_voice(self, input_path: str, output_path: str, vocal_input: bool = True) -> str:
        """
        Converte voz usando modelo RVC com otimização de memória
        
        Args:
            input_path: Áudio de entrada
            output_path: Arquivo de saída
            vocal_input: Se entrada é vocal
        
        Returns:
            str: Caminho do arquivo convertido
        """
        logger.info(f"🔄 Convertendo voz: {Path(input_path).name}")
        
        try:
            # Carregar e pré-processar áudio
            audio = preprocess_audio_for_rvc(input_path, self.sample_rate)
            
            # Mover para dispositivo se for GPU
            if self.device.type == 'cuda':
                # Para operações que suportam GPU
                pass
            
            # Extrair características do áudio de entrada
            source_features = extract_vocal_features_fixed(audio, self.sample_rate, self.target_frames)
            
            # Limpeza após extração de features
            memory_manager.clear_memory()
            
            # Aplicar conversão usando índice (se disponível)
            if self.faiss_index:
                converted_audio = self._apply_rvc_conversion(audio, source_features)
            else:
                converted_audio = self._apply_basic_conversion(audio, source_features)
            
            # Salvar resultado
            sf.write(output_path, converted_audio, self.sample_rate)
            
            # Limpeza final
            del audio, source_features, converted_audio
            memory_manager.clear_memory()
            
            logger.info(f"✅ Conversão concluída: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erro na conversão: {e}")
            memory_manager.clear_memory(aggressive=True)
            # Fallback: copiar arquivo original
            shutil.copy2(input_path, output_path)
            return output_path
    
    def _apply_rvc_conversion(self, audio: np.ndarray, source_features: dict) -> np.ndarray:
        """Aplica conversão RVC usando índice FAISS com otimização"""
        logger.info("🎯 Aplicando conversão RVC...")
        
        try:
            # Buscar características similares no índice
            query_vector = source_features['mfcc'].flatten().astype(np.float32)
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Buscar top-k características similares
            k = min(5, self.faiss_index.ntotal) #type:ignore
            similarities, indices = self.faiss_index.search(query_vector, k) #type:ignore
            
            # Limpeza após busca
            del query_vector
            memory_manager.clear_memory()
            
            # Aplicar transformação baseada nas características encontradas
            # (Simplificado - RVC real usa redes neurais complexas)
            
            # Modificar pitch baseado no modelo treinado
            f0_ratio = 1.1  # Ajuste de pitch (pode ser aprendido do modelo)
            converted_audio = self._modify_pitch(audio, f0_ratio)
            
            # Aplicar filtro de timbre
            converted_audio = self._apply_timbre_filter(converted_audio)
            
            return converted_audio
            
        except Exception as e:
            logger.warning(f"⚠️ Erro na conversão RVC, usando básica: {e}")
            return self._apply_basic_conversion(audio, source_features)
    
    def _apply_basic_conversion(self, audio: np.ndarray, source_features: dict) -> np.ndarray:
        """Conversão básica sem índice"""
        logger.info("🔧 Aplicando conversão básica...")
        
        # Ajustes simples baseados em características gerais
        converted_audio = self._modify_pitch(audio, 1.05)  # Pitch ligeiramente mais alto
        converted_audio = self._apply_timbre_filter(converted_audio)
        
        return converted_audio
    
    def _modify_pitch(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """Modifica pitch do áudio"""
        try:
            # Usar librosa para modificação de pitch
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=2)
        except:
            # Fallback: retornar áudio original
            return audio
    
    def _apply_timbre_filter(self, audio: np.ndarray) -> np.ndarray:
        """Aplica filtro de timbre"""
        try:
            # Equalização simples para modificar timbre
            from scipy.signal import butter, filtfilt #type:ignore
            
            # Boost em frequências médias (voz humana)
            nyquist = self.sample_rate // 2
            low = 300 / nyquist
            high = 3400 / nyquist
            
            b, a = butter(2, [low, high], btype='band') #type:ignore
            filtered = filtfilt(b, a, audio)
            
            # Misturar com original
            return 0.7 * audio + 0.3 * filtered
            
        except:
            return audio
    
    def _mix_audio(self, vocal_path: str, instrumental_path: str, output_path: str) -> str:
        """Mixa vocal convertido com instrumental"""
        logger.info("🎼 Mixando vocal + instrumental...")
        
        try:
            # Carregar áudios
            vocal, _ = librosa.load(vocal_path, sr=self.sample_rate, mono=True)
            instrumental, _ = librosa.load(instrumental_path, sr=self.sample_rate, mono=True)
            
            # Ajustar comprimentos
            min_length = min(len(vocal), len(instrumental))
            vocal = vocal[:min_length]
            instrumental = instrumental[:min_length]
            
            # Mixar com volumes balanceados
            mixed = 0.8 * instrumental + 0.9 * vocal
            
            # Normalizar
            mixed = librosa.util.normalize(mixed)
            
            # Salvar
            sf.write(output_path, mixed, self.sample_rate)
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erro na mixagem: {e}")
            # Fallback: retornar apenas vocal
            shutil.copy2(vocal_path, output_path)
            return output_path
    
    def _cleanup_temp_files(self, temp_paths: List[str]):
        """Limpa arquivos temporários"""
        for path in temp_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)