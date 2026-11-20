from manim_imports_ext import *


CLAIRE_ASSETS_DIRECTORY = os.path.join(
    os.path.dirname(manim_config.file_writer.output_directory),
    "Claire_Fruit_Basket_Assets"
)


class FruitBaskets(InteractiveScene):
    def construct(self):
        # Add the family
        family = ImageMobject(
            os.path.join(
                CLAIRE_ASSETS_DIRECTORY,
                "family.png"
            )
        )
        self.add(family)
