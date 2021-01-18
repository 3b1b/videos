from manim_imports_ext import *


class PrimePiEPttern(Scene):
    def construct(self):
        self.add(FullScreenFadeRectangle(fill_color=WHITE, fill_opacity=1))

        tex0 = Tex(
            "\\frac{1}{1^2}", "+"
            "\\frac{1}{2^2}", "+"
            "\\frac{1}{3^2}", "+"
            "\\frac{1}{4^2}", "+"
            "\\frac{1}{5^2}", "+"
            "\\frac{1}{6^2}", "+"
            # "\\frac{1}{7^2}", "+"
            # "\\frac{1}{8^2}", "+"
            # "\\frac{1}{9^2}", "+"
            # "\\frac{1}{10^2}", "+"
            # "\\frac{1}{11^2}", "+"
            # "\\frac{1}{12^2}", "+"
            "\\cdots",
            "=",
            "\\frac{\\pi^2}{6}",
        )
        self.alter_tex(tex0)
        # self.add(tex0)

        tex1 = Tex(
            "\\underbrace{\\frac{1}{1^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{2^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{3^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{4^2}}_{\\times (1 / 2)}", "+",
            "\\underbrace{\\frac{1}{5^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{6^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{7^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{8^2}}_{\\times (1 / 3)}", "+",
            "\\underbrace{\\frac{1}{9^2}}_{\\times (1 / 2)}", "+",
            "\\underbrace{\\frac{1}{10^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{11^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{12^2}}_{\\text{kill}}", "+",
            "\\cdots",
            # "=",
            # "\\frac{\\pi^2}{6}"
        )
        self.alter_tex(tex1)
        tex1.set_color_by_tex("kill", RED)
        tex1.set_color_by_tex("keep", GREEN_E)
        tex1.set_color_by_tex("times", BLUE_D)

        self.add(tex1)
        return

        # tex1 = Tex(
        #     "\\underbrace{\\frac{1}{1}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{-1}{3}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{5}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{7}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{9}}_{\\times (1 / 2)}", "+",
        #     "\\underbrace{\\frac{-1}{11}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{13}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{15}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{1}{17}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{19}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{21}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{-1}{23}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{25}}_{\\times (1 / 2)}", "+",
        #     "\\underbrace{\\frac{-1}{27}}_{\\times (1 / 3)}", "+",
        #     "\\cdots",
        #     "=",
        #     "\\frac{\\pi}{4}"
        # )
        # self.alter_tex(tex1)
        # VGroup(
        #     tex1[2 * 0],
        #     tex1[2 * 7],
        #     tex1[2 * 10],
        # ).set_color(RED)
        # VGroup(
        #     tex1[2 * 1],
        #     tex1[2 * 2],
        #     tex1[2 * 3],
        #     tex1[2 * 5],
        #     tex1[2 * 6],
        #     tex1[2 * 8],
        #     tex1[2 * 9],
        #     tex1[2 * 11],
        # ).set_color(GREEN_E)
        # VGroup(
        #     tex1[2 * 4],
        #     tex1[2 * 12],
        #     tex1[2 * 13],
        # ).set_color(BLUE_D)

        # self.add(tex1)

        # tex2 = Tex(
        #     "\\frac{-1}{3}", "+",
        #     "\\frac{1}{5}", "+",
        #     "\\frac{-1}{7}", "+",
        #     "\\frac{1}{2}", "\\cdot", "\\frac{1}{9}", "+",
        #     "\\frac{-1}{11}", "+",
        #     "\\frac{1}{13}", "+",
        #     "\\frac{1}{17}", "+",
        #     "\\frac{-1}{19}", "+",
        #     "\\frac{-1}{23}", "+",
        #     "\\frac{1}{2}", "\\cdot", "\\frac{1}{25}", "+",
        #     "\\frac{1}{3}", "\\cdot", "\\frac{-1}{27}", "+",
        #     "\\cdots",
        # )
        # self.alter_tex(tex2)
        # VGroup(
        #     tex2[2 * 0],
        #     tex2[2 * 1],
        #     tex2[2 * 2],
        #     tex2[2 * 5],
        #     tex2[2 * 6],
        #     tex2[2 * 7],
        #     tex2[2 * 8],
        #     tex2[2 * 9],
        # ).set_color(GREEN_E)
        # VGroup(
        #     tex2[2 * 3],
        #     tex2[2 * 4],
        #     tex2[2 * 10],
        #     tex2[2 * 11],
        #     tex2[2 * 12],
        #     tex2[2 * 13],
        # ).set_color(BLUE_D)

        tex2 = Tex(
            "\\frac{1}{2^2}", "+",
            "\\frac{1}{3^2}", "+",
            "\\frac{1}{2}", "\\cdot", "\\frac{1}{4^2}", "+",
            "\\frac{1}{5^2}", "+",
            "\\frac{1}{7^2}", "+",
            "\\frac{1}{3}", "\\cdot", "\\frac{1}{8^2}", "+",
            "\\frac{1}{2}", "\\cdot", "\\frac{1}{9^2}", "+",
            "\\frac{1}{11^2}", "+",
            "\\frac{1}{13^2}", "+",
            "\\frac{1}{4}", "\\cdot", "\\frac{1}{16^2}", "+",
            "\\cdots",
        )
        self.alter_tex(tex2)
        VGroup(
            tex2[2 * 0],
            tex2[2 * 1],
            tex2[2 * 4],
            tex2[2 * 5],
            tex2[2 * 10],
            tex2[2 * 11],
        ).set_color(GREEN_E)
        VGroup(
            tex2[2 * 2],
            tex2[2 * 3],
            tex2[2 * 6],
            tex2[2 * 7],
            tex2[2 * 8],
            tex2[2 * 9],
            tex2[2 * 12],
            tex2[2 * 13],
        ).set_color(BLUE_D)
        self.add(tex2)

        exp = Tex(
            "e^{\\left(",
            "0" * 30,
            "\\right)}",
            "= \\frac{\\pi^2}{6}"
        )
        self.alter_tex(exp)
        exp[1].set_opacity(0)
        tex2.replace(exp[1], dim_to_match=0)

        self.add(exp, tex2)

    def alter_tex(self, tex):
        tex.set_color(BLACK)
        tex.set_stroke(BLACK, 0, background=True)
        tex.set_width(FRAME_WIDTH - 1)
