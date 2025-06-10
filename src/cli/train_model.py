
model_name = "lucas"
training_audio_paths = [
    "downloads/2019 EM UMA MÚSICA.mp3",
    "downloads/2020 EM UMA MÚSICA.mp3", 
    "downloads/2021 EM UMA MÚSICA.mp3",
    "downloads/2022 EM UMA MÚSICA.mp3",
    "downloads/2023 EM UMA MÚSICA.mp3"
]

config = {
    'vocal_input': False,
    'epochs': 100,        
    'sample_rate': 40000 
}

from src.rvc import RVCModelTrainer, VoiceManager

trainer = RVCModelTrainer(model_name, config['sample_rate'])

# Treinar modelo com limpeza automática
model_path, index_path = trainer.train_model(
    audio_paths=training_audio_paths,
    vocal_input=config['vocal_input'],
    epochs=config['epochs']
)

print(f"\n🎉 Modelo RVC criado com sucesso!")
print(f"📄 Modelo: {model_path}")
print(f"🔍 Índice: {index_path}")
