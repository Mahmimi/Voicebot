import os
import unicodedata
import torch
import torchaudio
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier


# Load TTS model
checkpoint = '/mnt/nas/asr/tts/speecht5_5k_new_tok_fn1/checkpoint-5000'
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
processor = SpeechT5Processor.from_pretrained(checkpoint)
tokenizer = processor.tokenizer

def get_thai_characters():
    thai_characters = set()
    for code_point in range(0x0E01, 0x0E5B):  # Thai script Unicode range
        character = chr(code_point)
        if 'THAI' in unicodedata.name(character, ''):
            thai_characters.add(character)
    return list(thai_characters)

thai_characters_set = get_thai_characters()
tokenizer.add_tokens(thai_characters_set) 

# Load Vocoder model
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load Speaker Embedding model
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name, 
    run_opts={"device": device}, 
    savedir=os.path.join("/tmp", spk_model_name)
)

# Function to create speaker embeddings for waveform
def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(waveform)
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

audio, sampling_rate = torchaudio.load('/mnt/nas/asr/tts/datasets/test/tsync2_noon_99_271.wav')

# Create speaker embeddings for the dummy waveform
speaker_embeddings = create_speaker_embedding(audio)

# Generate speech with TTS model and speaker embeddings
# text = "ก า ร แ ข่ ง ขั น ฟุ ต บ อ ล โ ล ก"
# inputs = processor(text=text, return_tensors="pt")
# spectrogram = model.generate_speech(inputs["input_ids"], torch.tensor([speaker_embeddings]))

from speecht5_thai_tokenizer import SpeechT5OpenjtalkTokenizer
from transformers import SpeechT5FeatureExtractor

checkpoint = "/mnt/nas/asr/tts/speecht5_5k_new_tok/checkpoint-1500"
processor = SpeechT5Processor.from_pretrained(checkpoint)
tokenizer = processor.tokenizer

feature_extractor = SpeechT5FeatureExtractor.from_pretrained(checkpoint)
processor = SpeechT5Processor(feature_extractor, tokenizer)
tokenizer_instance = SpeechT5OpenjtalkTokenizer(vocab_file="vocab.json")

input = "สารพัด บอร์ด คมนาคม ลา ออก พรึ่บ"
input_ids = tokenizer_instance._tokenize(input)
label_numbers = [tokenizer_instance._convert_token_to_id(token) for token in input_ids]
input_ids = torch.tensor(label_numbers)
# input_ids = torch.tensor([tokenizer_instance._convert_token_to_id(token) for token in input_ids])

# Ensure that input_ids is a 2D tensor (batch size = 1)
input_ids = input_ids.unsqueeze(0)

# Move input_ids to the appropriate device
input_ids = input_ids.to(model.device)

print('input_ids=', input_ids)

spectrogram = model.generate_speech(input_ids, torch.tensor([speaker_embeddings]))

# Generate waveform with the vocoder
with torch.no_grad():
    speech = vocoder(spectrogram)

# Save the output waveform
import soundfile as sf
sf.write("output_fn1_5kcp.wav", speech.numpy(), samplerate=16000)
