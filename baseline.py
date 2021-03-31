from os import listdir, mkdir
from os.path import join, exists

from tqdm import tqdm

import shutil

import os

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc,
)

from natasha.doc import DocSpan

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

if os.path.exists(f"../baseline"):
    shutil.rmtree(f"../baseline")

mkdir(f"../baseline")
mkdir(f"../baseline/generic")
mkdir(f"../baseline/named")

for part in ["generic", "named"]:
    texts = {}
    anns = {}

    files = os.listdir(f"../data/public_test/{part}")

    for file in files:
        name = file[:-4]

        if file[-3:] == "txt":
            text = open(f"../data/public_test/{part}/{file}", encoding='utf-8').read()

            texts[name] = text
        elif file[-3:] == "ann":
            ann = open(f"../data/public_test/{part}/{file}", encoding='utf-8').read().strip().split('\n')

            anns[name] = ann

    for name in tqdm(texts):
        text = texts[name]

        ann = anns[name]

        f = open(f"../baseline/{part}/{name}.norm", 'w', encoding='utf-8')

        for line in ann:
            spans = list(map(int, line.strip().split()))
            entry = ''
            while spans:
                start, stop = spans[0], spans[1]
                entry += text[start:stop] + " "

                spans = spans[2:]

            entry = entry.strip()

            doc = Doc(entry)

            doc.segment(segmenter)

            doc.tag_morph(morph_tagger)
            doc.parse_syntax(syntax_parser)
            doc.tag_ner(ner_tagger)

            found = False
            span = None
            for s in doc.spans:
                if s.text == entry:
                    span = s
                    found = True
                    break

            if not found:
                span = DocSpan(
                    start=0
                    , stop=len(entry)
                    , type='ORG'
                    , text=entry
                    , tokens=[token for token in doc.tokens]
                )
            if span is not None:
                span.normalize(morph_vocab)

                f.write(f"{span.normal}\n")

        f.close()
