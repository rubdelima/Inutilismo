#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import re

sys.path.append(str(Path(__file__).parent.parent))

from utils.youtube import YouTubeDownloader

def is_valid_youtube_url(url: str) -> bool:
    """Verifica se a URL é válida do YouTube"""
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtube\.com/playlist\?list=[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/channel/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/@[\w-]+',
    ]
    
    return any(re.match(pattern, url) for pattern in youtube_patterns)

def interactive_mode():
    print("🎵 LVCAS YouTube Downloader")
    print("=" * 40)
    print("1. Download de vídeo")
    print("2. Download de áudio apenas") 
    print("3. Download de playlist")
    print("4. Download múltiplos vídeos")
    print("0. Sair")
    
    downloader = YouTubeDownloader()
    
    while True:
        print("\n" + "-" * 40)
        choice = input("Escolha uma opção (0-4): ").strip()
        
        if choice == "0":
            print("👋 Até logo!")
            break
        
        if choice not in ["1", "2", "3", "4"]:
            print("❌ Opção inválida!")
            continue
        
        url = input("Digite a URL: ").strip()
        if not url:
            print("❌ URL não pode estar vazia!")
            continue
        
        # Validar URL
        if not is_valid_youtube_url(url):
            print("❌ URL do YouTube inválida!")
            print("   Formatos aceitos:")
            print("   - https://www.youtube.com/watch?v=...")
            print("   - https://youtu.be/...")
            print("   - https://www.youtube.com/playlist?list=...")
            continue
        
        # Qualidade
        quality = input("Qualidade (Enter=padrão, max, 1080p, 720p, etc): ").strip() or None
        if quality and quality != "max":
            try:
                quality = int(quality.replace('p', ''))
            except:
                quality = None
        
        if choice == "1":  # Vídeo
            print(f"📥 Iniciando download do vídeo...")
            success = downloader.download_video(url, quality=quality)
            print("✅ Concluído!" if success else "❌ Falha")
            
        elif choice == "2":  # Áudio
            print(f"🎵 Iniciando download do áudio...")
            success = downloader.download_video(url, quality=quality, audio_only=True)
            print("✅ Concluído!" if success else "❌ Falha")
            
        elif choice == "3":  # Playlist
            workers = input("Workers (Enter=padrão): ").strip()
            max_workers = int(workers) if workers.isdigit() else None
            
            print(f"📋 Iniciando download da playlist...")
            results = downloader.download_playlist(url, quality=quality, max_workers=max_workers)
            
            if 'error' in results:
                print(f"❌ Erro: {results['error']}")
            else:
                print(f"📊 Sucessos: {results.get('successful_downloads', 0)}")
                print(f"❌ Falhas: {results.get('failed_downloads', 0)}")
            
        elif choice == "4":  # Múltiplos
            print("Digite URLs separadas por vírgula:")
            urls = [u.strip() for u in url.split(',') if u.strip()]
            
            # Validar todas as URLs
            invalid_urls = [u for u in urls if not is_valid_youtube_url(u)]
            if invalid_urls:
                print(f"❌ URLs inválidas encontradas: {len(invalid_urls)}")
                for invalid in invalid_urls[:3]:  # Mostrar apenas 3
                    print(f"   - {invalid}")
                continue
            
            if len(urls) < 2:
                print("❌ Para múltiplos vídeos, separe as URLs com vírgula")
                continue
            
            workers = input("Workers (Enter=padrão): ").strip()
            max_workers = int(workers) if workers.isdigit() else None
            
            print(f"📥 Iniciando download de {len(urls)} vídeos...")
            results = downloader.download_multiple_videos(urls, quality=quality, max_workers=max_workers)
            print(f"📊 Sucessos: {results['successful_downloads']}")
            print(f"❌ Falhas: {results['failed_downloads']}")

def main():
    parser = argparse.ArgumentParser(description="YouTube Downloader LVCAS")
    parser.add_argument('url', nargs='?', help='URL do vídeo/playlist')
    parser.add_argument('-q', '--quality', help='Qualidade (720p, 1080p, max)')
    parser.add_argument('-a', '--audio-only', action='store_true', help='Apenas áudio')
    parser.add_argument('-p', '--playlist', action='store_true', help='Playlist')
    parser.add_argument('-w', '--workers', type=int, help='Workers paralelos')
    
    args = parser.parse_args()
    
    if not args.url:
        interactive_mode()
        return
    
    # Validar URL na linha de comando também
    if not is_valid_youtube_url(args.url):
        print("❌ URL do YouTube inválida!")
        return
    
    downloader = YouTubeDownloader()
    
    if args.playlist:
        results = downloader.download_playlist(args.url, quality=args.quality, max_workers=args.workers)
        if 'error' in results:
            print(f"❌ Erro: {results['error']}")
        else:
            print(f"Sucessos: {results.get('successful_downloads', 0)}, Falhas: {results.get('failed_downloads', 0)}")
    else:
        success = downloader.download_video(args.url, quality=args.quality, audio_only=args.audio_only)
        print("✅ Sucesso!" if success else "❌ Falha")

if __name__ == "__main__":
    main()