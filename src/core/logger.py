import logging
import os
from typing import Optional
from src.core.config import GLOBAL_CONFIG

class ColoredFormatter(logging.Formatter):
    """Formatter para adicionar cores aos logs no console"""
    
    # Códigos de cores ANSI
    COLORS = {
        'DEBUG': '\033[36m',      # Ciano
        'INFO': '\033[32m',       # Verde
        'WARNING': '\033[33m',    # Amarelo
        'ERROR': '\033[31m',      # Vermelho
        'CRITICAL': '\033[35m',   # Magenta
    }
    TIME_COLOR = '\033[90m'       # Cinza para timestamp
    NAME_COLOR = '\033[94m'       # Azul para nome
    RESET = '\033[0m'             # Reset cor
    
    def format(self, record):
        # Aplicar cores
        log_color = self.COLORS.get(record.levelname, self.RESET)
        colored_levelname = f"{log_color}{record.levelname}{self.RESET}"
        colored_time = f"{self.TIME_COLOR}%(asctime)s{self.RESET}"
        colored_name = f"{self.NAME_COLOR}%(name)s{self.RESET}"
        
        # Criar formato com cores
        colored_format = f"{colored_time} : {colored_name} : {colored_levelname} : %(message)s"
        
        # Aplicar formato temporariamente
        original_format = self._style._fmt
        self._style._fmt = colored_format
        
        formatted = super().format(record)
        
        # Restaurar formato original
        self._style._fmt = original_format
        
        return formatted

def get_logger(name: str = "", console: bool = True) -> logging.Logger:
    """Logger simples para o projeto LVCAS
    
    Args:
        name: Nome do módulo/componente (opcional)
        console: Se deve também logar no console
    """
    
    # Criar diretório de logs se não existir
    os.makedirs('logs', exist_ok=True)
    
    # Configurar logger
    logger_name = name if name is not None else ''
    logger = logging.getLogger(logger_name)
    
    # Limpar handlers existentes para evitar duplicação
    if logger.handlers:
        logger.handlers.clear()
    
    # Configurar nível como DEBUG para capturar tudo
    logger.setLevel(logging.DEBUG)
    
    # Evitar propagação para o logger root
    logger.propagate = False
    
    # Handler para arquivo geral - sempre salva em general.log
    general_file_path = 'logs/general.log'
    general_file_handler = logging.FileHandler(general_file_path, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
    general_file_handler.setFormatter(file_formatter)
    general_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(general_file_handler)
    
    # Handler para arquivo específico (se name foi fornecido)
    if name:
        specific_file_path = f'logs/{name}.log'
        specific_file_handler = logging.FileHandler(specific_file_path, encoding='utf-8')
        specific_file_handler.setFormatter(file_formatter)
        specific_file_handler.setLevel(logging.DEBUG)
        logger.addHandler(specific_file_handler)
    
    # Handler para console - verificar configuração global
    try:
        show_console = console or GLOBAL_CONFIG.get("logging", {}).get("global_verbose", True)
    except (KeyError, AttributeError):
        # Fallback se config não existir
        show_console = console
    
    if show_console:
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
    
    return logger