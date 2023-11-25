from mindcraft.infra.embeddings.embeddings import Embeddings


class LTM_embeddings:
    def __init__(self, name: Embeddings):
        switch = {
            Embeddings.MINILM: 'all-MiniLM-L6-v2'
        }

        self.embeddings_name = switch.get(name)
