#transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

#thai2transformers
from thai2transformers.preprocess import process_transformers
from thai2transformers.metrics import (
    classification_metrics,
    multilabel_classification_metrics,
)
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
    SEFR_SPLIT_TOKEN
)

class DialogModelConfig():
    def __init__(self, model_name:str = 'wangchanberta-base-att-spm-uncased') -> None:
        self.model_names = [
            'wangchanberta-base-att-spm-uncased',
            'xlm-roberta-base',
            'bert-base-multilingual-cased',
            'wangchanberta-base-wiki-newmm',
            'wangchanberta-base-wiki-ssg',
            'wangchanberta-base-wiki-sefr',
            'wangchanberta-base-wiki-spm',
        ]
        self.tokenizers = {
            'wangchanberta-base-att-spm-uncased': AutoTokenizer,
            'xlm-roberta-base': AutoTokenizer,
            'bert-base-multilingual-cased': AutoTokenizer,
            'wangchanberta-base-wiki-newmm': ThaiWordsNewmmTokenizer,
            'wangchanberta-base-wiki-ssg': ThaiWordsSyllableTokenizer,
            'wangchanberta-base-wiki-sefr': FakeSefrCutTokenizer,
            'wangchanberta-base-wiki-spm': ThaiRobertaTokenizer,
        }
        self.public_models = ['xlm-roberta-base', 'bert-base-multilingual-cased']
        self.model_name = model_name

        #create tokenizer
        self.tokenizer = self.tokenizers[model_name].from_pretrained(
                        f'airesearch/{model_name}' if model_name not in self.public_models else f'{model_name}',
                        revision='main',
                        model_max_length=416,)
        
        #pipeline
        self.zero_classify = pipeline(task='zero-shot-classification',
                tokenizer=self.tokenizer,
                model=AutoModelForSequenceClassification.from_pretrained(
                    f'airesearch/{model_name}' if model_name not in self.public_models else f'airesearch/{model_name}-finetuned',
                    revision='finetuned@xnli_th')
                )
        
    