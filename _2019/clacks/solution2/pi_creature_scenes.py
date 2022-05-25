from manim_imports_ext import *


class OnAnsweringTwice(TeacherStudentsScene):
    def construct(self):
        question = TexText("Why $\\pi$?")
        question.move_to(self.screen)
        question.to_edge(UP)
        other_questions = VGroup(
            TexText("Frequency of collisions?"),
            TexText("Efficient simulation?"),
            TexText("Time until last collision?"),
        )
        for mob in other_questions:
            mob.move_to(self.hold_up_spot, DOWN)

        self.add(question)

        self.student_says(
            "But we already \\\\ solved it",
            bubble_config={"direction": LEFT},
            target_mode="raise_left_hand",
            added_anims=[self.teacher.change, "thinking"]
        )
        self.play_student_changes("sassy", "angry")
        self.wait()
        self.play(
            RemovePiCreatureBubble(self.students[2]),
            self.change_students("erm", "erm"),
            ApplyMethod(
                question.move_to, self.hold_up_spot, DOWN,
                path_arc=-90 * DEGREES,
            ),
            self.teacher.change, "raise_right_hand",
        )
        shown_questions = VGroup(question)
        for oq in other_questions:
            self.play(
                shown_questions.shift, 0.85 * UP,
                FadeInFromDown(oq),
                self.change_students(
                    *["pondering"] * 3,
                    look_at=oq
                )
            )
            shown_questions.add(oq)
        self.wait(3)


class AskAboutEqualMassMomentumTransfer(TeacherStudentsScene):
    def construct(self):
        self.student_says("Why?")
        self.play_student_changes("confused", "confused")
        self.wait()
        self.play(
            RemovePiCreatureBubble(self.students[2]),
            self.teacher.change, "raise_right_hand"
        )
        self.play_all_student_changes("pondering")
        self.look_at(self.hold_up_spot + 2 * UP)
        self.wait(5)


class ComplainAboutRelevanceOfAnalogy(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            "Why would \\\\ you care",
            target_mode="maybe"
        )
        self.play_student_changes(
            "angry", "sassy", "maybe",
            added_anims=[self.teacher.change, "guilty"]
        )
        self.wait(2)
        self.play(
            self.teacher.change, "raise_right_hand",
            self.change_students(
                "pondering", "erm", "pondering",
                look_at=self.hold_up_spot,
            ),
            RemovePiCreatureBubble(self.students[2])
        )
        self.play(
            self.students[2].change, "thinking",
            self.hold_up_spot + UP,
        )
        self.wait(3)


class ReplaceOneTrickySceneWithAnother(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            "This replaces one tricky\\\\problem with another",
            index=1,
            target_mode="sassy",
            added_anims=[self.teacher.change, "happy"],
        )
        self.play_student_changes("erm", "sassy", "angry")
        self.wait(4)
        self.play(
            RemovePiCreatureBubble(self.students[1]),
            self.teacher.change, "raise_right_hand",
            self.change_students(*3 * ["pondering"])
        )
        self.look_at(self.hold_up_spot + 2 * UP)
        self.wait(5)


class NowForTheGoodPart(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            r"Now for the \\ good part!",
            target_mode="hooray",
            added_anims=[self.change_students(
                "hooray", "surprised", "happy"
            )],
        )
        self.wait(2)
