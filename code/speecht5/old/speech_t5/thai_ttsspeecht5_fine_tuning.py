import os
import gc
import torch
import torchaudio
import unicodedata
import pandas as pd
import pyarrow as pa
from datasets import Dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from speechbrain.pretrained import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5HifiGan, SpeechT5ForTextToSpeech, SpeechT5FeatureExtractor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
tokenizer = processor.tokenizer

# def get_thai_characters():
#     thai_characters = set()
#     for code_point in range(0x0E01, 0x0E5B):  # Thai script Unicode range
#         character = chr(code_point)
#         if 'THAI' in unicodedata.name(character, ''):
#             thai_characters.add(character)
#     return list(thai_characters)

# thai_characters_set = get_thai_characters()
# tokenizer.add_tokens(thai_characters_set) 

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams( 
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

from torchaudio.transforms import Resample
def prepare_dataset(batch):
    # Load audio file
    audio, sampling_rate = torchaudio.load(batch["path"])

    # Check if the sampling rate is not 16000
    if sampling_rate != 16000:
        transform = Resample(orig_freq=sampling_rate, new_freq=16000)
        waveform = transform(audio)
        waveform = waveform[0]
    else:
        # Extract the one-dimensional waveform
        waveform = audio[0]
    
    # Process the text and audio using the SpeechT5Processor
    batch = processor(
        text=batch["spaced_text"],
        audio_target=waveform,
        sampling_rate=16000, 
        return_attention_mask=False,
    )
    
    # Strip off the batch dimension
    batch["labels"] = batch["labels"][0]

    # Use SpeechBrain to obtain x-vector
    batch["speaker_embeddings"] = create_speaker_embedding(audio)

    return batch

df = pd.read_excel('TSynC2_Nun.xlsx', usecols=['path','spaced_text', 'duration'], engine='openpyxl')
df['path'] = '/mnt/nas/asr/tts/datasets/test/' + df['path']
df = df[df['duration'] < 30]
df = df[df['duration'] > 5]

df2 = pd.read_excel('cv_female4.xlsx', usecols=['path','spaced_text', 'duration'], engine='openpyxl')
# df2 = df2[df2['spaced_text'].str.contains(' ')]
df = pd.concat([df, df2], ignore_index=True)
# df = df.iloc[2000:2100]
df = df.reset_index(drop=True)

print(df.tail(5))
# os._exit()

# from pythainlp.tokenize import word_tokenize
# df['spaced_text'] = df['spaced_text'].str.replace(' ', '')
# df['spaced_text'] = df['spaced_text'].apply(lambda x: ' '.join(word_tokenize(x, engine='icu')))

# Split the DataFrame into train and test sets (80:20 split)
split_ratio = 0.8
train_size = int(split_ratio * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

ds_train = Dataset(pa.Table.from_pandas(train_df))
ds_test = Dataset(pa.Table.from_pandas(test_df))

ds_train = ds_train.map(prepare_dataset)
ds_test = ds_test.map(prepare_dataset)

from speecht5_thai_tokenizer import SpeechT5OpenjtalkTokenizer
feature_extractor = SpeechT5FeatureExtractor.from_pretrained(checkpoint)
processor = SpeechT5Processor(feature_extractor, tokenizer)
tokenizer_instance = SpeechT5OpenjtalkTokenizer(vocab_file="vocab.json")

from tqdm import tqdm

def tokenize_and_convert(example, tokenizer_instance, error_indices):
    original_text = example.get("spaced_text", "")
    index_key = example.get("__index_level_0__", None)

    try:
        # Tokenize and convert the text
        input_ids = tokenizer_instance._tokenize(original_text)
        label_numbers = [tokenizer_instance._convert_token_to_id(token) for token in input_ids]

        # Update the "input_ids" column in the example
        example["input_ids"] = label_numbers
    except Exception as e:
        # Handle the exception (you might want to log or print the error)
        print(f"Error processing example at index {index_key}: {original_text}")
        print(f"Error details: {e}")

        # Add the index to the list of error indices
        error_indices.append(index_key)

    return example

# Initialize an empty list to store error indices
error_indices_train = []
error_indices_test = []

# Apply the function to both training and test datasets
ds_train = ds_train.map(lambda x: tokenize_and_convert(x, tokenizer_instance, error_indices_train))
ds_test = ds_test.map(lambda x: tokenize_and_convert(x, tokenizer_instance, error_indices_test))

# Print error indices
print("Error indices in training set:", error_indices_train)
print("Error indices in test set:", error_indices_test)

# Remove examples with errors from the training set
ds_train = ds_train.filter(lambda example, idx: idx not in error_indices_train, with_indices=True)

# Remove examples with errors from the test set
ds_test = ds_test.filter(lambda example, idx: idx not in error_indices_test, with_indices=True)

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
            target_lengths = target_lengths.new(
                [length - length % model.config.reduction_factor for length in target_lengths]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
model.config.use_cache = False 
model.resize_token_embeddings(len(tokenizer))

name = 'speecht5_10k_tsycn&cv'
training_args = Seq2SeqTrainingArguments(
    output_dir= f"./{name}", 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    auto_find_batch_size=True,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=5000,
    fp16=True,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to="tensorboard",
    load_best_model_at_end=True,
    save_total_limit=3,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    data_collator=data_collator,
    tokenizer=processor,
)

def main():
    print("model.num_parameters = ", model.num_parameters())
    with open(rf"./{name}/trainer_setting.txt", "w") as f:
        print(trainer.args, file=f)
        f.close()
    trainer.train()
    trainer.save_model()
    trainer.save_state()

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    main()