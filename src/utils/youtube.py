import yt_dlp
import os
import yaml
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import threading
from src.core.logger import get_logger

class YouTubeDownloader:
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Carregar config primeiro
        self.config = self._load_config(config_path)
        
        # Depois inicializar logger
        self.logger = get_logger('youtube_downloader')
        
        # Setup
        self.output_path = self.config['youtube'].get('download_path', './downloads')
        os.makedirs(self.output_path, exist_ok=True)
        self._progress_bars = {}
        self._progress_lock = threading.Lock()
        
        self.logger.info("YouTube Downloader inicializado")
    
    def _load_config(self, config_path: str) -> dict:
        """Carrega config sem dependência de logger"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            # Config padrão se não conseguir carregar
            return {
                'youtube': {
                    'max_video_quality': 1080,
                    'max_audio_quality': 'max',
                    'audio_codec': 'mp3',
                    'download_path': './downloads',
                    'filename_template': '%(title)s.%(ext)s',
                    'playlist': {
                        'concurrent_downloads': 3,
                        'delay_between_downloads': 1
                    }
                }
            }
    
    def _get_format_selector(self, quality: Any, audio_only: bool = False) -> str:
        """
        Seletor de formato para garantir melhor qualidade de áudio
        """
        if audio_only:
            if quality == "max":
                # Priorizar melhor bitrate disponível em AAC/Opus
                return 'bestaudio[acodec^=aac]/bestaudio[acodec^=opus]/bestaudio'
            else:
                return f'bestaudio[abr<={quality}][acodec^=aac]/bestaudio[abr<={quality}][acodec^=opus]/bestaudio[abr<={quality}]/bestaudio'
        else:
            # Para vídeos, sempre pegar o melhor áudio disponível
            if quality == "max":
                return 'bestvideo[height>=1080]+bestaudio[acodec^=aac]/bestvideo[height>=1080]+bestaudio[acodec^=opus]/bestvideo[height>=1080]+bestaudio/bestvideo+bestaudio/best[height>=1080]/best'
            elif isinstance(quality, int):
                return f'bestvideo[height<={quality}]+bestaudio[acodec^=aac]/bestvideo[height<={quality}]+bestaudio[acodec^=opus]/bestvideo[height<={quality}]+bestaudio/bestvideo[height<={quality}]/best[height<={quality}]/best'
            else:
                return 'bestvideo+bestaudio[acodec^=aac]/bestvideo+bestaudio[acodec^=opus]/bestvideo+bestaudio/best'
    
    def _progress_hook(self, d: Dict[str, Any]):
        if d['status'] == 'finished':
            filename = d.get('filename', 'arquivo')
            print(f"✅ Concluído: {os.path.basename(filename)}")
            self.logger.info(f"Download concluído: {filename}")
                
        elif d['status'] == 'error':
            error_msg = d.get('error', 'Erro desconhecido')
            self.logger.error(f"Erro no download: {error_msg}")
    
    def download_video(self, url: str, quality: Optional[Any] = None, audio_only: bool = False) -> bool:
        youtube_config = self.config['youtube']
        
        if quality is None:
            quality = youtube_config.get('max_video_quality', 1080)
        
        quality_str = f"áudio {quality}" if audio_only else f"vídeo {quality}"
        self.logger.info(f"Iniciando download: {url} (qualidade: {quality_str})")
        print(f"🎯 Qualidade solicitada: {quality_str}")
        
        # Configurar formato
        format_selector = self._get_format_selector(quality, audio_only)
        print(f"🔧 Seletor: {format_selector}")
        self.logger.info(f"Seletor de formato: {format_selector}")
        
        ydl_opts = {
            'outtmpl': os.path.join(self.output_path, youtube_config.get('filename_template', '%(title)s.%(ext)s')),
            'progress_hooks': [self._progress_hook],
            'quiet': True,
            'no_warnings': True,
            'format': format_selector,
            # IMPORTANTE: Forçar merge de vídeo+áudio
            'merge_output_format': 'mp4',
        }
        
        if audio_only:
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': youtube_config.get('audio_codec', 'mp3'),
                'preferredquality': '256',
            }]
        else:
            # Para vídeos, forçar conversão do áudio para AAC
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }, {
                'key': 'FFmpegFixupM4a',
            }]
            # Forçar re-codificação se necessário
            ydl_opts['postprocessor_args'] = {
                'ffmpeg': ['-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k']
            }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extrair info primeiro para debug
                info = ydl.extract_info(url, download=False)
                
                # Debug: mostrar formatos de vídeo disponíveis
                formats = info.get('formats', [])
                video_formats = [f for f in formats if f.get('height') and f.get('vcodec') != 'none']
                available_heights = sorted(list(set([f.get('height') for f in video_formats])), reverse=True)
                
                # Debug: mostrar formatos de áudio disponíveis
                audio_formats = [f for f in formats if f.get('acodec') and f.get('acodec') != 'none' and not f.get('vcodec')]
                audio_info = []
                for f in audio_formats:
                    codec = f.get('acodec', 'unknown')
                    bitrate = f.get('abr', 'N/A')
                    audio_info.append(f"{codec}@{bitrate}kbps")
                
                print(f"📊 Qualidades de vídeo: {available_heights}")
                print(f"🔊 Qualidades de áudio: {audio_info}")
                self.logger.info(f"Qualidades disponíveis - Vídeo: {available_heights}, Áudio: {audio_info}")
                
                # Fazer download
                ydl.download([url])
            
            self.logger.info(f"Download concluído com sucesso: {url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no download: {e}")
            print(f"❌ Erro: {str(e)}")
            return False
    
    def download_playlist(self, playlist_url: str, quality: Optional[Any] = None, max_workers: Optional[int] = None) -> Dict[str, Any]:
        if max_workers is None:
            max_workers = self.config['youtube'].get('playlist', {}).get('concurrent_downloads', 3)
        
        self.logger.info(f"Iniciando download de playlist: {playlist_url}")
        
        try:
            extract_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'playlist_items': None,
            }
            
            with yt_dlp.YoutubeDL(extract_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
            
            if 'entries' not in playlist_info:
                print("ℹ️  URL é de vídeo único, fazendo download direto.")
                success = self.download_video(playlist_url, quality)
                return {
                    'total_videos': 1,
                    'successful_downloads': 1 if success else 0,
                    'failed_downloads': 0 if success else 1,
                    'failed_urls': [] if success else [playlist_url]
                }
            
            entries = [entry for entry in playlist_info.get('entries', []) if entry is not None]
            playlist_title = playlist_info.get('title', 'Playlist')
            
            print(f"🎵 Playlist: {playlist_title}")
            print(f"📊 Total: {len(entries)} vídeos")
            
            self.logger.info(f"Playlist extraída: {playlist_title} com {len(entries)} vídeos")
            
            if len(entries) == 0:
                print("❌ Nenhum vídeo encontrado na playlist")
                return {'error': 'Nenhum vídeo encontrado'}
            
            results = {
                'total_videos': len(entries),
                'successful_downloads': 0,
                'failed_downloads': 0,
                'failed_urls': []
            }
            
            playlist_progress = tqdm(total=len(entries), desc=f"🎵 {playlist_title[:30]}...", unit="vídeo")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {}
                for entry in entries:
                    if entry and entry.get('id'):
                        video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                        future = executor.submit(self.download_video, video_url, quality)
                        future_to_url[future] = video_url
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        success = future.result()
                        if success:
                            results['successful_downloads'] += 1
                        else:
                            results['failed_downloads'] += 1
                            results['failed_urls'].append(url)
                    except Exception as e:
                        self.logger.error(f"Erro no download paralelo: {e}")
                        results['failed_downloads'] += 1
                        results['failed_urls'].append(url)
                    
                    playlist_progress.update(1)
                    time.sleep(self.config['youtube'].get('playlist', {}).get('delay_between_downloads', 1))
            
            playlist_progress.close()
            print(f"\n✅ Concluído! Sucessos: {results['successful_downloads']}, Falhas: {results['failed_downloads']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na playlist: {e}")
            print(f"❌ Erro na playlist: {str(e)}")
            return {'error': str(e)}
    
    def download_playlist_audio(self, playlist_url: str, quality: Optional[Any] = None, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Download apenas áudio de uma playlist"""
        if max_workers is None:
            max_workers = self.config['youtube'].get('playlist', {}).get('concurrent_downloads', 3)
        
        self.logger.info(f"Iniciando download de áudio da playlist: {playlist_url}")
        
        try:
            extract_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'playlist_items': None,
            }
            
            with yt_dlp.YoutubeDL(extract_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
            
            if 'entries' not in playlist_info:
                print("ℹ️  URL é de vídeo único, fazendo download de áudio direto.")
                success = self.download_video(playlist_url, quality, audio_only=True)
                return {
                    'total_videos': 1,
                    'successful_downloads': 1 if success else 0,
                    'failed_downloads': 0 if success else 1,
                    'failed_urls': [] if success else [playlist_url]
                }
            
            entries = [entry for entry in playlist_info.get('entries', []) if entry is not None]
            playlist_title = playlist_info.get('title', 'Playlist')
            
            print(f"🎵 Playlist (Áudio): {playlist_title}")
            print(f"📊 Total: {len(entries)} áudios")
            
            self.logger.info(f"Playlist de áudio extraída: {playlist_title} com {len(entries)} áudios")
            
            if len(entries) == 0:
                print("❌ Nenhum áudio encontrado na playlist")
                return {'error': 'Nenhum áudio encontrado'}
            
            results = {
                'total_videos': len(entries),
                'successful_downloads': 0,
                'failed_downloads': 0,
                'failed_urls': []
            }
            
            playlist_progress = tqdm(total=len(entries), desc=f"🎵 {playlist_title[:30]}... (Áudio)", unit="áudio")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {}
                for entry in entries:
                    if entry and entry.get('id'):
                        video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                        future = executor.submit(self.download_video, video_url, quality, True)  # audio_only=True
                        future_to_url[future] = video_url
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        success = future.result()
                        if success:
                            results['successful_downloads'] += 1
                        else:
                            results['failed_downloads'] += 1
                            results['failed_urls'].append(url)
                    except Exception as e:
                        self.logger.error(f"Erro no download de áudio paralelo: {e}")
                        results['failed_downloads'] += 1
                        results['failed_urls'].append(url)
                    
                    playlist_progress.update(1)
                    time.sleep(self.config['youtube'].get('playlist', {}).get('delay_between_downloads', 1))
            
            playlist_progress.close()
            print(f"\n✅ Concluído! Sucessos: {results['successful_downloads']}, Falhas: {results['failed_downloads']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na playlist de áudio: {e}")
            print(f"❌ Erro na playlist de áudio: {str(e)}")
            return {'error': str(e)}

    def download_playlist_with_options(self, playlist_url: str, quality: Optional[Any] = None, audio_only: bool = False, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Download de playlist com opção de áudio apenas ou vídeo+áudio"""
        if audio_only:
            return self.download_playlist_audio(playlist_url, quality, max_workers)
        else:
            return self.download_playlist(playlist_url, quality, max_workers)
    
    def download_multiple_videos(self, urls: List[str], quality: Optional[Any] = None, max_workers: Optional[int] = None) -> Dict[str, Any]:
        if max_workers is None:
            max_workers = 3
        
        self.logger.info(f"Download múltiplo iniciado: {len(urls)} URLs")
        
        results = {'total_videos': len(urls), 'successful_downloads': 0, 'failed_downloads': 0, 'failed_urls': []}
        
        progress = tqdm(total=len(urls), desc="📥 Vídeos", unit="vídeo")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.download_video, url, quality): url for url in urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success = future.result()
                    if success:
                        results['successful_downloads'] += 1
                    else:
                        results['failed_downloads'] += 1
                        results['failed_urls'].append(url)
                except Exception as e:
                    self.logger.error(f"Erro no download múltiplo: {e}")
                    results['failed_downloads'] += 1
                    results['failed_urls'].append(url)
                
                progress.update(1)
        
        progress.close()
        return results