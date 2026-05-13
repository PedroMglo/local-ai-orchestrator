"""Tests for ComplexityClassifier."""

from orchestrator.context.base import Complexity
from orchestrator.core.complexity import HeuristicComplexityClassifier


class TestComplexityClassifier:
    def setup_method(self):
        self.clf = HeuristicComplexityClassifier()

    # -- SIMPLE --
    def test_simple_short(self):
        assert self.clf.classify("o que é DNS?") == Complexity.NORMAL  # 4 words → normal

    def test_simple_two_words(self):
        assert self.clf.classify("porta PostgreSQL") == Complexity.SIMPLE

    def test_simple_three_words(self):
        assert self.clf.classify("DNS port?") == Complexity.SIMPLE

    # -- NORMAL --
    def test_normal_medium(self):
        assert self.clf.classify("diferença entre git rebase e merge") == Complexity.NORMAL

    def test_normal_question(self):
        assert self.clf.classify("como funciona o Docker?") == Complexity.NORMAL

    # -- COMPLEX --
    def test_complex_code_gen(self):
        assert self.clf.classify("implementa um servidor HTTP em Python com async") == Complexity.COMPLEX

    def test_complex_long(self):
        # Contains " e " (boolean) + >10 words → DEEP
        assert self.clf.classify("como configuro o Qdrant para funcionar com replicas e sharding em modo distribuído") == Complexity.DEEP

    def test_complex_without_boolean(self):
        assert self.clf.classify("como configuro o Qdrant para funcionar com replicas, sharding, modo distribuído") == Complexity.COMPLEX

    # -- DEEP --
    def test_deep_analysis(self):
        assert self.clf.classify("analisa os prós e contras de microservices vs monólito") == Complexity.DEEP

    def test_deep_multi_question(self):
        assert self.clf.classify("o que é melhor? monólito ou microservices? e porquê?") == Complexity.DEEP

    def test_deep_debug(self):
        assert self.clf.classify("analisa este erro e explica passo a passo como o resolver") == Complexity.DEEP
