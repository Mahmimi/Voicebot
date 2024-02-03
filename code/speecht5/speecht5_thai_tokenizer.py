import json
import logging
import os
from pathlib import Path
import re
from transformers import SpeechT5Tokenizer
from transformers.models.speecht5.tokenization_speecht5 import (
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
)
from itertools import chain
from typing import List, Optional
from pythainlp.transliterate import romanize

logger = logging.getLogger(__name__)

NP_CHARCTERS = " !\"#$%&'()=~|`{+*}<>?_-^\\@[;:],./　！”＃＄％＆’（）＝～｜｀｛＋＊｝＜＞？＿ー＾￥＠「；：」、。・`"

def _g2p_with_np_thai(text: str, np_list: str) -> List[str]:
    np_pattern = re.compile(f"([{re.escape(np_list)}])")

    return list(
        chain.from_iterable(
            [
                list(romanize(text, engine="royin")) if text not in np_list else [text]
                for text in np_pattern.split(text)
                if len(text) > 0
            ]
        )
    )

# def _g2p_with_np_thai(text: str, np_list: str) -> List[str]:
#     np_pattern = re.compile(f"([{re.escape(np_list)}])")

#     romanized_text = romanize(text, engine="royin")

#     # Ensure the Romanized version has the same length as the original text
#     if len(romanized_text) < len(text):
#         romanized_text += " " * (len(text) - len(romanized_text))
#     elif len(romanized_text) > len(text):
#         romanized_text = romanized_text[:len(text)]

#     return list(
#         chain.from_iterable(
#             [
#                 [char] if char in np_list else [romanized_text[i]]
#                 for i, char in enumerate(np_pattern.split(text))
#                 if len(char) > 0
#             ]
#         )
#     )

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "/mnt/nas/asr/tts/speecht5_5k_new_tok/checkpoint-1500": "/mnt/nas/asr/tts/vocab.json",
    },
}

class SpeechT5OpenjtalkTokenizer(SpeechT5Tokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        non_phenome_characters: str = NP_CHARCTERS,
        **kwargs,
    ):
        try:
            super().__init__(
                vocab_file=None,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                **kwargs,
            )
        except TypeError:
            pass

        self.non_phenome_characters = non_phenome_characters
        self.vocab_file = vocab_file
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sep_token = "</s>"
        self.cls_token = "<cls>"
        self.mask_token = "<mask>"
        self.additional_special_tokens = "<ctc_blank>"

        self._load_vocab()

    def _load_vocab(self):
        if isinstance(self.vocab_file, str) and self.vocab_file.endswith(".json"):
            with open(self.vocab_file, encoding="utf-8") as f:
                self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}

    @property
    # def bos_token_id(self) -> int | None:
    #     return super().bos_token_id
    def bos_token_id(self) -> Optional[int]:
        return super().bos_token_id

    @property
    # def vocab_size(self):
    #     return len(self.label2id)

    # def get_vocab(self):
    #     return self.label2id

    def __getstate__(self):
        state = super().__getstate__()
        del state["sp_model"]
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self._load_vocab()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ):
        if filename_prefix is None:
            filename_prefix = ".json"

        save_path = Path(save_directory)
        if not save_path.is_dir():
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        vocab_path = Path(save_directory) / Path(f"vocab{filename_prefix}")
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)

        return (str(vocab_path),)

    def _tokenize(self, text: str) -> List[str]:
        return _g2p_with_np_thai(text, self.non_phenome_characters)

    def _convert_token_to_id(self, token):
        return self.label2id.get(token, self.label2id.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.id2label.get(index, self.unk_token)

# # Create an instance of SpeechT5OpenjtalkTokenizer
# tokenizer_instance = SpeechT5OpenjtalkTokenizer(vocab_file="vocab.json")

# # Tokenize
# # text_to_tokenize = "深夜に一緒に勉強しませんか ?"
# text_to_tokenize = "สวัสดีครับ ทดสอบภาษาไทย"
# tokens = tokenizer_instance._tokenize(text_to_tokenize)
# print("Tokenized Text:", tokens)

# # Convert token to ID
# token_id = tokenizer_instance._convert_token_to_id(tokens[0])
# print("Token ID:", token_id)

# # Convert ID to token
# converted_token = tokenizer_instance._convert_id_to_token(token_id)
# print("Converted Token:", converted_token)

# from transformers import SpeechT5Processor

# # Specify the pre-trained model checkpoint
# checkpoint = "microsoft/speecht5_tts"

# # Load the processor and tokenizer
# processor = SpeechT5Processor.from_pretrained(checkpoint)
# tokenizer = processor.tokenizer

# # Example: Tokenize and prepare data for training
# text_to_tokenize = "สารพัด บอร์ด คมนาคม ลา ออก พรึ่บ"  # Thai text

# # Tokenize the text
# # input_ids = tokenizer(text_to_tokenize, return_tensors="pt").input_ids

# tokenizer_instance = SpeechT5OpenjtalkTokenizer(vocab_file="vocab.json")
# input_ids = tokenizer_instance._tokenize(text_to_tokenize)
# print("Tokenized Text:", input_ids)

# token_lst = [tokenizer_instance._convert_token_to_id(token) for token in input_ids]
# print('token_lst', token_lst)

# # Convert label numbers to tokens
# tokens = [tokenizer_instance._convert_id_to_token(label_number) for label_number in token_lst]
# # Print the list of tokens
# print("Tokens:", tokens)

# [4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 2]
