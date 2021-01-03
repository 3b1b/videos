from manimlib.scene.scene import Scene


class ExternallyAnimatedScene(Scene):
    def construct(self):
        raise Exception("Don't actually run this class.")
