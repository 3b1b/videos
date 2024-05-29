from manim_imports_ext import *
from _2024.transformers.helpers import *


class BasicMLPWalkThrough(InteractiveScene):
    def construct(self):
        self.set_floor_plane("xz")

        # Show two matrices

        # Sequence of embeddings comes in
        embedding_array = EmbeddingArray()
        self.add(embedding_array)

        # Multiply by the up-projection

        # Compactify the matrix multiplication notation

        # Add a bias

        # ReLU

        # Describe these as neurons? Draw the classic image?

        # Down projection

        # Again with a bias

        # Add it to the original

        # Show it done in parallel to all embeddings
        pass


## Maybe not needed?
class ShowBiasBakedIntoWeightMatrix(InteractiveScene):
    def construct(self):
        pass

