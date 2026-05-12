from datasets import load_dataset
import spacy
from collections import Counter


class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.dataset = load_dataset("bentrevett/multi30k")[split]
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.src_vocab = None
        self.tgt_vocab = None

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.dataset:
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])
            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)

        self.src_vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.tgt_vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}

        for token, freq in src_counter.items():
            if token not in self.src_vocab:
                self.src_vocab[token] = len(self.src_vocab)

        for token, freq in tgt_counter.items():
            if token not in self.tgt_vocab:
                self.tgt_vocab[token] = len(self.tgt_vocab)

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        data = []

        for example in self.dataset:
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])

            src_indices = [
                self.src_vocab.get(tok, self.src_vocab["<unk>"])
                for tok in src_tokens
            ]

            tgt_indices = (
                [self.tgt_vocab["<sos>"]] +
                [self.tgt_vocab.get(tok, self.tgt_vocab["<unk>"]) for tok in tgt_tokens] +
                [self.tgt_vocab["<eos>"]]
            )

            data.append((src_indices, tgt_indices))

        return data