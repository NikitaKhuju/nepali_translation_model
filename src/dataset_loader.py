import os
import datasets

from typing import Union, List, Optional

_HOMEPAGE = "https://github.com/facebookresearch/flores"

_LICENSE = "CC-BY-SA-4.0"


_LANGUAGES = [ "npi_Deva", "eng_Latn"]
_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

_SPLITS = ["dev", "devtest"]

_SENTENCES_PATHS = {
    lang: {
        split: os.path.join("flores200_dataset", split, f"{lang}.{split}")
        for split in _SPLITS
    } for lang in _LANGUAGES
}

_METADATA_PATHS = {
    split: os.path.join("flores200_dataset", f"metadata_{split}.tsv")
    for split in _SPLITS
}

from itertools import permutations

def _pairings(iterable, r=2):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p


class Flores200Config(datasets.BuilderConfig):
    def __init__(self, lang: str, lang2: str = None, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.lang = lang
        self.lang2 = lang2


class Flores200(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        Flores200Config(
            name=lang,
            description=f"FLORES-200: {lang} subset.",
            lang=lang
        )
        for lang in _LANGUAGES
    ] +  [
        Flores200Config(
            name="all",
            description=f"FLORES-200: all language pairs",
            lang=None
        )
    ] +  [
        Flores200Config(
            name=f"{l1}-{l2}",
            description=f"FLORES-200: {l1}-{l2} aligned subset.",
            lang=l1,
            lang2=l2
        ) for (l1,l2) in _pairings(_LANGUAGES)
    ]

    def _info(self):
        features = {
            "id": datasets.Value("int32"),
            "URL": datasets.Value("string"),
            "domain": datasets.Value("string"),
            "topic": datasets.Value("string"),
            "has_image": datasets.Value("int32"),
            "has_hyperlink": datasets.Value("int32")
        }
        if self.config.name != "all" and "-" not in self.config.name:
            features["sentence"] = datasets.Value("string")
        elif "-" in self.config.name:
            for lang in [self.config.lang, self.config.lang2]:
                features[f"sentence_{lang}"] = datasets.Value("string")
        else:
            for lang in _LANGUAGES:
                features[f"sentence_{lang}"] = datasets.Value("string")
        return datasets.DatasetInfo(
            features=datasets.Features(features),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )
    
    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)

        def _get_sentence_paths(split):
            if isinstance(self.config.lang, str) and isinstance(self.config.lang2, str):
                sentence_paths = [os.path.join(dl_dir, _SENTENCES_PATHS[lang][split]) for lang in (self.config.lang, self.config.lang2)]
            elif isinstance(self.config.lang, str):
                sentence_paths = os.path.join(dl_dir, _SENTENCES_PATHS[self.config.lang][split])
            else:
                sentence_paths = [os.path.join(dl_dir, _SENTENCES_PATHS[lang][split]) for lang in _LANGUAGES]
            return sentence_paths
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "sentence_paths": _get_sentence_paths(split),
                    "metadata_path": os.path.join(dl_dir, _METADATA_PATHS[split]),
                }
            ) for split in _SPLITS
        ]

    def _generate_examples(self, sentence_paths: Union[str, List[str]], metadata_path: str, langs: Optional[List[str]] = None):
        """Yields examples as (key, example) tuples."""
        if isinstance(sentence_paths, str):
            with open(sentence_paths, "r") as sentences_file:
                with open(metadata_path, "r") as metadata_file:
                    metadata_lines = [l.strip() for l in metadata_file.readlines()[1:]]
                    for id_, (sentence, metadata) in enumerate(
                        zip(sentences_file, metadata_lines)
                    ):
                        sentence = sentence.strip()
                        metadata = metadata.split("\t")
                        yield id_, {
                            "id": id_ + 1,
                            "sentence": sentence,
                            "URL": metadata[0],
                            "domain": metadata[1],
                            "topic": metadata[2],
                            "has_image": 1 if metadata == "yes" else 0,
                            "has_hyperlink": 1 if metadata == "yes" else 0
                        }
        else:
            sentences = {}
            if len(sentence_paths) == len(_LANGUAGES):
                langs = _LANGUAGES
            else:
                langs = [self.config.lang, self.config.lang2]
            for path, lang in zip(sentence_paths, langs):
                with open(path, "r") as sent_file:
                    sentences[lang] = [l.strip() for l in sent_file.readlines()]
            with open(metadata_path, "r") as metadata_file:
                metadata_lines = [l.strip() for l in metadata_file.readlines()[1:]]
            for id_, metadata in enumerate(metadata_lines):
                metadata = metadata.split("\t")
                yield id_, {
                    **{
                        "id": id_ + 1,
                        "URL": metadata[0],
                        "domain": metadata[1],
                        "topic": metadata[2],
                        "has_image": 1 if metadata == "yes" else 0,
                        "has_hyperlink": 1 if metadata == "yes" else 0
                    }, **{
                        f"sentence_{lang}": sentences[lang][id_]
                        for lang in langs
                    }
                }