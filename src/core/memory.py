import gc
import torch
import psutil
from typing import Optional

from src.core.logger import get_logger
from src.core.models import RAMInfo, VRAMInfo

class AdvancedMemoryManager:
    """Gerenciador avançado de memória VRAM/RAM para RVC"""
    def __init__(self, vram_limit_percent=0.85, ram_limit_percent=0.80, verbose=False):
        self.vram_limit = vram_limit_percent
        self.ram_limit = ram_limit_percent
        self.model_cache = {}
        self.logger = get_logger("memory_manager", console=verbose)
        
    def get_memory_status(self):
        """Retorna status completo da memória"""
        status = {
            'ram': self._get_ram_info(),
            'vram': self._get_vram_info() if torch.cuda.is_available() else None
        }
        return status
    
    def _get_ram_info(self)->RAMInfo:
        """Informações da RAM"""
        ram = psutil.virtual_memory()
        return RAMInfo(
            total = ram.total / 1e9,
            available = ram.available / 1e9,
            percent = ram.percent,
            free = ram.free / 1e9
        )
    
    def _get_vram_info(self) -> Optional[VRAMInfo]:
        """Informações da VRAM"""
        if not torch.cuda.is_available():
            return None
            
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        cached = torch.cuda.memory_reserved(0)
        
        return VRAMInfo(
            total= total / 1e9,
            allocated = allocated / 1e9,
            cached = cached / 1e9,
            free = (total - allocated) / 1e9,
            percent = (allocated / total) * 100
        )
    
    def should_use_gpu(self, estimated_vram_gb=1.0):
        """Decide se deve usar GPU baseado na memória disponível"""
        if not torch.cuda.is_available():
            return False
            
        vram_info = self._get_vram_info()
        
        assert vram_info is not None
        
        if vram_info.free < estimated_vram_gb:
            return False
            
        if vram_info.percent > (self.vram_limit * 100):
            return False
            
        return True
    
    def clear_memory(self, aggressive=False):
        """Limpa memória VRAM e RAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("🗑️ Limpando cache da VRAM")
        
        self.logger.info("🗑️ Limpando Memória com Garbage Collector")
        gc.collect()
        
        if aggressive:
            if len(self.model_cache) > 3:
                oldest_key = next(iter(self.model_cache))
                del self.model_cache[oldest_key]
                self.logger.info("🗑️ Cache de modelo removido")
    
    def smart_device_selection(self, model_size_gb=1.0):
        """Seleção inteligente de dispositivo"""
        if self.should_use_gpu(model_size_gb):
            return torch.device('cuda')
        else:
            ram_info = self._get_ram_info()
            if ram_info.percent > (self.ram_limit * 100):
                self.logger.info("⚠️ Memória RAM alta, liberando cache...")
                self.clear_memory(aggressive=True)
            return torch.device('cpu')
    
    def monitor_memory(self):
        """Monitor de memória em tempo real"""
        status = self.get_memory_status()
        
        self.logger.info(f"💾 RAM: {status['ram'].percent:.1f}% ({status['ram'].available:.1f}GB livres)")
        
        if status['vram']:
            self.logger.info(f"🎮 VRAM: {status['vram'].percent:.1f}% ({status['vram'].free:.1f}GB livres)")