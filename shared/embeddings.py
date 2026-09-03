from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer

class EmbeddingClass(ABC):

    def __init__(self, modelname):
        self.modelname = modelname
    
    @abstractmethod
    def encode(self, corpora: list|str):
        pass

class TokenizerClass(ABC):

    def __init__(self, modelname):
        self.modelname = modelname
    
    @abstractmethod
    def tokenize(self, corpora: list|str):
        pass


class SentenceTransformerEmbedding(EmbeddingClass):

    def __init__(self, modelname: str = "sentence-transformers/all-MiniLM-L6-v2"):

        self.modelname = modelname
        self.model = SentenceTransformer(self.modelname, device="cpu")
        self.embedding_dimension = self.model.get_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length
    
    def encode(self, corpus):

        return self.model.encode(corpus)

class SentenceTransformerTokenizer(TokenizerClass):

    def __init__(self, modelname: str = "sentence-transformers/all-MiniLM-L6-v2"):

        self.modelname = modelname
        self.model = SentenceTransformer(self.modelname, device="cpu")
    
    def tokenize(self, corpus):

        return self.model.preprocess(corpus)["input_ids"]
