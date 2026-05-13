"""Tests for IntentClassifier."""

from orchestrator.context.base import Intent
from orchestrator.core.intent import HeuristicIntentClassifier


class TestIntentClassifier:
    def setup_method(self):
        self.clf = HeuristicIntentClassifier()

    # -- GENERAL --
    def test_general_question(self):
        assert self.clf.classify("o que é DNS?") == Intent.GENERAL

    def test_general_english(self):
        assert self.clf.classify("what is a linked list?") == Intent.GENERAL

    def test_general_definition(self):
        assert self.clf.classify("explica o protocolo HTTP") == Intent.GENERAL

    # -- LOCAL --
    def test_local_my_notes(self):
        assert self.clf.classify("resume as minhas notas sobre Python") == Intent.LOCAL

    def test_local_vault(self):
        assert self.clf.classify("o que tenho no vault sobre Docker?") == Intent.LOCAL

    def test_local_files(self):
        assert self.clf.classify("mostra os meus ficheiros de config") == Intent.LOCAL

    # -- CODE --
    def test_code_function(self):
        assert self.clf.classify("escreve uma função Python para ler CSV") == Intent.CODE

    def test_code_refactor(self):
        assert self.clf.classify("refactora para async") == Intent.CODE

    def test_code_debug(self):
        assert self.clf.classify("debug este traceback") == Intent.CODE

    # -- SYSTEM --
    def test_system_ram(self):
        assert self.clf.classify("quanta RAM tenho livre?") == Intent.SYSTEM

    def test_system_gpu(self):
        assert self.clf.classify("qual é a temperatura da GPU?") == Intent.SYSTEM

    def test_system_disk(self):
        assert self.clf.classify("quanto espaço em disco tenho?") == Intent.SYSTEM

    # -- System false positives --
    def test_machine_learning_not_system(self):
        assert self.clf.classify("explica machine learning") == Intent.GENERAL

    def test_system_design_not_system(self):
        assert self.clf.classify("o que é system design?") == Intent.GENERAL

    # -- GRAPH --
    def test_graph_architecture(self):
        assert self.clf.classify("qual é a arquitectura do meu projeto?") in (
            Intent.LOCAL_AND_GRAPH, Intent.GRAPH
        )

    def test_graph_dependencies(self):
        assert self.clf.classify("o que depende do meu módulo de config?") in (
            Intent.LOCAL_AND_GRAPH, Intent.GRAPH
        )

    # -- COMBINED --
    def test_system_and_local(self):
        assert self.clf.classify("tenho RAM suficiente para os meus modelos instalados?") == Intent.SYSTEM_AND_LOCAL

    def test_local_and_graph(self):
        assert self.clf.classify("qual o fluxo do meu pipeline de sync?") == Intent.LOCAL_AND_GRAPH
