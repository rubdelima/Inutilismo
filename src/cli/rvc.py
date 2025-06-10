from src.rvc import VoiceManager

model_path = './models/lucas/lucas.pth'
index_path = './models/lucas/lucas.index'

voice_manager = VoiceManager(model_path, index_path)

audio_path = "./data/2023/Ryan Gosling - I'm Just Ken.flac" 
output_path = "mtm_test_vocal_only.wav"
vocal_input = False
only_vocal = True

vocal_output = voice_manager.mtm(
    audio_path, output_path , vocal_input, only_vocal
)

full_output = voice_manager.mtm(
    audio_path, "mtm_test_full.wav" , vocal_input, False
)