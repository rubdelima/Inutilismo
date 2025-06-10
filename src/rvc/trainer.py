from typing import List
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import faiss # type:ignore
import numpy as np
import torch
import soundfile as sf # type:ignore
import shutil
import librosa
from typing import Optional

from src.audio import (
    separate_vocals_advanced, 
    preprocess_audio_for_rvc, 
    extract_vocal_features_fixed,
)
from src.core import AdvancedMemoryManager, NUM_WORKERS, get_logger

memory_manager = AdvancedMemoryManager()

logger = get_logger("rvc_trainer", console=False)

class RVCModelTrainer:
    """Trainer para modelos RVC com limpeza automática"""
    
    def __init__(self, model_name: str, target_sr: int = 40000):
        self.model_name = model_name
        self.target_sr = target_sr
        self.model_dir = Path("models") / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Diretórios temporários
        self.temp_dir = self.model_dir / "temp"
        self.dataset_dir = self.temp_dir / "dataset"
        
        self.target_frames = 200
        self.num_workers = NUM_WORKERS
        
        # Configurar dispositivo com base na memória disponível
        self.device = memory_manager.smart_device_selection(model_size_gb=1.5)
        
        logger.info(f"🎯 RVC Trainer inicializado: {model_name}")
        logger.info(f"📁 Diretório: {self.model_dir}")
        logger.info(f"🔄 Workers: {self.num_workers}")
        logger.info(f"🎮 Dispositivo: {self.device}")
        
        # Monitor inicial de memória
        memory_manager.monitor_memory()
    
    def train_model(self, audio_paths: List[str], vocal_input: bool = False, epochs: int = 100) -> tuple[str, str]:
        """
        Treina modelo RVC e mantém apenas arquivos essenciais
        
        Args:
            audio_paths: Lista de caminhos de áudio
            vocal_input: Se áudios já são vocais isolados
            epochs: Número de épocas
            
        Returns:
            tuple: (model_path, index_path)
        """
        try:
            logger.info(f"🚀 Iniciando treinamento com {len(audio_paths)} arquivos")
            
            # Limpeza inicial de memória
            memory_manager.clear_memory(aggressive=True)
            
            # 1. Preparar dados temporários
            self._prepare_training_data(audio_paths, vocal_input)
            
            # Limpeza após preparação de dados
            memory_manager.clear_memory()
            memory_manager.monitor_memory()
            
            # 2. Extrair features e criar índice (processo que mais consome memória)
            index_path = self._create_faiss_index()
            
            # Limpeza após criação do índice
            memory_manager.clear_memory(aggressive=True)
            
            # 3. Criar arquivo do modelo
            model_path = self._create_model_file(epochs)
            
            # 4. Limpar arquivos temporários
            self._cleanup_temporary_files()
            
            # Limpeza final
            memory_manager.clear_memory()
            
            logger.info(f"✅ Modelo criado com sucesso!")
            logger.info(f"📄 Arquivos finais:")
            logger.info(f"  - {model_path}")
            logger.info(f"  - {index_path}")
            
            # Monitor final de memória
            memory_manager.monitor_memory()
            
            return str(model_path), str(index_path)
            
        except Exception as e:
            logger.error(f"❌ Erro durante o treinamento: {e}")
            # Limpar em caso de erro
            self._cleanup_temporary_files()
            memory_manager.clear_memory(aggressive=True)
            raise
    
    def _prepare_training_data(self, audio_paths: List[str], vocal_input: bool):
        """Prepara dados de treinamento temporariamente"""
        logger.info("🔄 Preparando dados de treinamento...")
        
        self.temp_dir.mkdir(exist_ok=True)
        self.dataset_dir.mkdir(exist_ok=True)
        
        max_length = 10 * self.target_sr  # 10 segundos por segmento
        
        for i, audio_path in enumerate(tqdm(audio_paths, desc="Processando áudios")):
            try:
                # Monitor de memória a cada arquivo
                if i % 2 == 0:  # A cada 2 arquivos
                    memory_manager.monitor_memory()
                    
                    # Verificar se precisa limpar memória
                    if not memory_manager.should_use_gpu(0.5):
                        logger.info("🧹 Limpando memória durante processamento...")
                        memory_manager.clear_memory()
                
                if not vocal_input:
                    # Separar vocal (temporariamente) - processo que consome memória
                    temp_sep_dir = self.temp_dir / f"separation_{i}"
                    logger.info(f"🎤 Separando vocal do arquivo {i+1}/{len(audio_paths)}")
                    vocals_path, _ = separate_vocals_advanced(audio_path, str(temp_sep_dir))
                    source_audio = vocals_path
                    
                    # Limpeza após separação vocal
                    memory_manager.clear_memory()
                else:
                    source_audio = audio_path
                
                # Processar áudio
                audio_data = preprocess_audio_for_rvc(source_audio, self.target_sr)
                
                # Quebrar em segmentos
                if len(audio_data) > max_length:
                    segments = [audio_data[i:i+max_length] 
                              for i in range(0, len(audio_data), max_length)]
                else:
                    segments = [audio_data]
                
                # Salvar segmentos
                for j, segment in enumerate(segments):
                    if len(segment) < self.target_sr:  # Ignorar muito curtos
                        continue
                    
                    output_path = self.dataset_dir / f"audio_{i:03d}_{j:02d}.wav"
                    sf.write(output_path, segment, self.target_sr)
                
                # Limpar separação temporária
                if not vocal_input:
                    shutil.rmtree(temp_sep_dir, ignore_errors=True)
                    
            except Exception as e:
                logger.warning(f"⚠️ Erro ao processar {audio_path}: {e}")
                # Limpar memória em caso de erro
                memory_manager.clear_memory()
                continue
    
    def _create_faiss_index(self) -> Path:
        """Cria índice FAISS a partir dos dados temporários"""
        logger.info("🔍 Extraindo features e criando índice...")
        
        audio_files = list(self.dataset_dir.glob("*.wav"))
        
        if not audio_files:
            raise ValueError("Nenhum arquivo de áudio processado encontrado")
        
        # Verificar se FAISS tem suporte para GPU
        has_gpu_support = hasattr(faiss, 'StandardGpuResources')
        use_gpu_faiss = has_gpu_support and memory_manager.should_use_gpu(2.0) and torch.cuda.is_available()
        
        if use_gpu_faiss:
            try:
                logger.info("🎮 Testando recursos GPU do FAISS...")
                # Configurar FAISS para GPU se disponível
                gpu_id = 0
                res = faiss.StandardGpuResources()
                logger.info("✅ FAISS GPU disponível")
            except Exception as e:
                logger.warning(f"⚠️ FAISS GPU não disponível: {e}")
                use_gpu_faiss = False
                res = None
                gpu_id = None
        else:
            if not has_gpu_support:
                logger.info("💻 FAISS instalado sem suporte GPU (usando CPU)")
            else:
                logger.info("💻 Usando CPU para FAISS (memória insuficiente ou CUDA indisponível)")
            res = None
            gpu_id = None
        
        # Extrair features sequencialmente (sem divisão em lotes)
        valid_features = []
        
        logger.info(f"🔍 Processando {len(audio_files)} arquivos sequencialmente...")
        
        for i, audio_file in enumerate(tqdm(audio_files, desc="Extraindo features")):
            # Monitor de memória apenas a cada 10 arquivos
            if i % 10 == 0:
                memory_manager.monitor_memory()
                
                # Verificar se precisa limpar memória
                if not memory_manager.should_use_gpu(0.5):
                    logger.info("🧹 Limpando memória durante processamento...")
                    memory_manager.clear_memory()
            
            mfcc_flat = self._extract_features_for_file(audio_file)
            
            if mfcc_flat is not None:
                valid_features.append(mfcc_flat)
            
            # Limpeza de memória a cada 20 arquivos
            if i % 20 == 0:
                memory_manager.clear_memory()
        
        if not valid_features:
            raise ValueError("Nenhuma feature válida extraída")
        
        # Normalizar tamanhos
        feature_sizes = [len(f) for f in valid_features]
        if len(set(feature_sizes)) > 1:
            min_size = min(feature_sizes)
            valid_features = [f[:min_size] for f in valid_features]
            logger.info(f"🔧 Features normalizadas para tamanho: {min_size}")
        
        # Criar matriz de features
        logger.info("🔄 Criando matriz de features...")
        features_matrix = np.vstack(valid_features).astype(np.float32)
        
        # Limpeza após criação da matriz
        del valid_features  # Liberar lista original
        memory_manager.clear_memory()
        
        # Limpar valores inválidos
        if np.any(np.isnan(features_matrix)) or np.any(np.isinf(features_matrix)):
            features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalizar
        faiss.normalize_L2(features_matrix)
        
        # Criar índice FAISS com GPU se disponível
        dimension = features_matrix.shape[1]
        logger.info(f"🏗️ Criando índice FAISS {dimension}D...")
        
        if use_gpu_faiss and res and gpu_id is not None:
            try:
                # Índice GPU
                logger.info("🎮 Criando índice na GPU...")
                cpu_index = faiss.IndexFlatIP(dimension)
                index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
                index.add(features_matrix)
                
                # Converter de volta para CPU para salvar
                cpu_index = faiss.index_gpu_to_cpu(index)
                index = cpu_index
                logger.info("✅ Índice criado na GPU e transferido para CPU")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao usar GPU, fallback para CPU: {e}")
                # Fallback para CPU
                index = faiss.IndexFlatIP(dimension)
                index.add(features_matrix)
                logger.info("✅ Índice criado na CPU (fallback)")
        else:
            # Índice CPU
            logger.info("💻 Criando índice na CPU...")
            index = faiss.IndexFlatIP(dimension)
            index.add(features_matrix)
            logger.info("✅ Índice criado na CPU")
        
        # Salvar índice
        index_path = self.model_dir / f"{self.model_name}.index"
        faiss.write_index(index, str(index_path))
        
        # Limpeza final
        del features_matrix
        if use_gpu_faiss and res:
            # Liberar recursos GPU se foram usados
            try:
                del res
            except:
                pass
        memory_manager.clear_memory(aggressive=True)
        
        logger.info(f"✅ Índice criado: {index.ntotal} vetores, {dimension}D")
        return index_path
    
    def _extract_features_for_file(self, audio_file: Path) -> Optional[np.ndarray]:
        """Extrai features de um arquivo com otimização de memória"""
        try:
            # Carregar áudio diretamente no dispositivo se for GPU
            audio, _ = librosa.load(audio_file, sr=self.target_sr, mono=True)
            
            # Se usando GPU, mover dados para GPU durante processamento
            if self.device.type == 'cuda':
                # Processamento otimizado para GPU seria aqui
                # Por enquanto, manter CPU para librosa
                pass
            
            features = extract_vocal_features_fixed(audio, self.target_sr, self.target_frames)
            
            # Liberar áudio da memória imediatamente
            del audio
            
            return features['mfcc'].flatten()
        except Exception as e:
            logger.warning(f"⚠️ Erro ao extrair features de {audio_file}: {e}")
            return None
    
    def _create_model_file(self, epochs: int) -> Path:
        """Cria arquivo do modelo"""
        logger.info("💾 Criando arquivo do modelo...")
        
        model_data = {
            'model_name': self.model_name,
            'sample_rate': self.target_sr,
            'target_frames': self.target_frames,
            'epochs': epochs,
            'version': '2.0',
            'type': 'rvc_simplified'
        }
        
        model_path = self.model_dir / f"{self.model_name}.pth"
        torch.save(model_data, model_path)
        
        return model_path
    
    def _cleanup_temporary_files(self):
        """Remove todos os arquivos temporários"""
        logger.info("🧹 Limpando arquivos temporários...")
        
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("✅ Arquivos temporários removidos")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao limpar temporários: {e}")
        
        # Remover outros arquivos desnecessários se existirem
        unnecessary_patterns = [
            "dataset_files.txt",
            "features_*.npz",
            "metadata.json"  # Será recriado automaticamente pelo VoiceManager
        ]
        
        for pattern in unnecessary_patterns:
            for file_path in self.model_dir.glob(pattern):
                try:
                    file_path.unlink()
                    logger.debug(f"🗑️ Removido: {file_path.name}")
                except Exception:
                    pass
