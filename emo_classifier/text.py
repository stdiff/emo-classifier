import pandas as pd
from en_core_web_md import load as spacy_load


class Tokenizer:
    def __init__(self, with_lemmatization: bool = False):
        if with_lemmatization:
            exclude = ["tok2vec", "parser", "ner"]
        else:
            exclude = ["tok2vec", "tagger", "parser", "attribute_ruler", "ner", "lemmatizer"]

        self.nlp = spacy_load(exclude=exclude)

    @staticmethod
    def doc2tokens(doc) -> list[str]:
        return [token.lower_ for token in doc if not token.is_stop and not token.is_punct]

    def __call__(self, text: str) -> list[str]:
        return self.doc2tokens(self.nlp(text))

    def tokenize_in_batch(self, s_text: pd.Series) -> pd.Series:
        """No significant improvement if with_lemmtatization is True"""
        return pd.Series(self.nlp.pipe(s_text), index=s_text.index, name="tokens").apply(self.doc2tokens)


if __name__ == "__main__":

    tokenizer = Tokenizer(with_lemmatization=False)
    s_text = pd.Series(["I am a cat.", "I don't have any name yet."], name="hoge")
    x = tokenizer.tokenize_in_batch(s_text)
    print(x)
