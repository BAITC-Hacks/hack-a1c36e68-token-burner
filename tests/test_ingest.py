from pathlib import Path

import docx
import pytest

from backend.ingest import ingest_file, parse_clauses

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def red8():
    return ingest_file(SAMPLES / "red8.pdf", "before", "before-1")


@pytest.fixture(scope="module")
def red9():
    return ingest_file(SAMPLES / "red9.pdf", "after", "after-1")


def ids(doc):
    return [c.clause_id for c in doc.clauses]


def test_samples_parse_completely(red8, red9):
    for doc in (red8, red9):
        assert doc.pages == 25
        assert doc.warnings == []
        assert len(ids(doc)) == len(set(ids(doc))), "clause ids must be unique"
        assert all(c.text for c in doc.clauses)
    assert len(red8.clauses) == 439
    assert len(red9.clauses) == 446


@pytest.mark.parametrize(
    "fixture, clause_id, page, starts_with",
    [
        ("red8", "3.4.а", 6, "Департамент непрерывного мониторинга системы внутреннего контроля (ДНМ)."),
        ("red9", "3.4.а", 6, "Департамент ИТ-аудита и анализа данных (ДИТААД)."),
        ("red9", "3.4.б", 6, "Департамент операционного аудита (ДОА)."),
        ("red8", "5.3.3.б", 8, "определение целей проверок, критериев, порядка оценки эффективности"),
        ("red9", "5.3.3.б", 8, "выявления рисков с недостаточным или дублирующим покрытием субъектами СВК"),
        ("red8", "5.6.2", 10, "формировать группы контроля качества с привлечением работников БВА"),
        ("red9", "5.6.2", 10, "использовать конфиденциальную информацию"),
        ("red8", "9.37", 19, "Главный аудитор несет общую ответственность за осуществление контроля"),
        ("red9", "9.37", 18, "Главный аудитор несет общую ответственность за осуществление контроля"),
    ],
)
def test_key_clauses(request, fixture, clause_id, page, starts_with):
    clause = request.getfixturevalue(fixture).clause(clause_id)
    assert clause is not None, clause_id
    assert clause.text.startswith(starts_with)
    assert clause.page == page


def test_clause_text_is_exact_and_bounded(red8, red9):
    assert red8.clause("9.37").text.endswith("функции Директору направления внутреннего аудита.")
    assert red9.clause("9.37").text.endswith("функции Директору операционного аудита.")
    assert red8.clause("5.6.2").text == (
        "формировать группы контроля качества с привлечением работников БВА в соответствии с ресурсным "
        "планом и бюджетом затрат БВА;"
    )
    assert red8.clause("5.3.3.б").parent_id == "5.3.3"


def test_empty_clause_is_kept(red8):
    assert red8.clause("5.5.3").text == ";"


def test_inline_numbers_and_headings(red8):
    assert red8.clause("3.9").text == "Рабочие места работников БВА могут располагаться в филиалах Общества."
    assert red8.clause("3.10").text.startswith("Работники могут выполнять")
    assert red8.clause("10").text == "Контроль качества"  # "... мероприятий. 10.Контроль качества"
    assert red8.clause("9.60.д").text.startswith("период времени")


def test_cross_references_dates_and_toc_are_not_clauses(red8):
    assert "5.8.1" in red8.clause("5.11.2").text  # "не противоречащей п. 5.8.1 и 5.8.2"
    assert "06.12.2011" in red8.clause("1.10").text
    assert ids(red8)[-2:] == ["14", "toc"]


def test_text_with_children(red9):
    text = red9.text_with_children("3.4")
    assert "ДИТААД" in text and "ДОА" in text and "ДНМ" in text


def test_txt_inline_numbering(tmp_path):
    path = tmp_path / "unit.txt"
    path.write_text(
        "1. Общие положения\n"
        "1.1. Отдел выполняет функции. 1.2.Отдел согласно п. 1.1. Положения ведет учет.\n"
        "1.3. отдел обязан:\n"
        "а. готовить отчеты;\n"
        "б. хранить документы.\n"
        "2. Права\n"
        "2.1. Запрашивать информацию от 01.02.2023.\n",
        encoding="utf-8",
    )
    doc = ingest_file(path, "before", "b1")
    assert ids(doc) == ["1", "1.1", "1.2", "1.3", "1.3.а", "1.3.б", "2", "2.1"]
    assert doc.clause("1.2").text == "Отдел согласно п. 1.1. Положения ведет учет."
    assert doc.clause("1.1").page is None


def test_docx_explicit_numbers(tmp_path):
    path = tmp_path / "unit.docx"
    d = docx.Document()
    for line in ["Положение об отделе", "1. Общие положения", "1.1. Отдел подчиняется директору.",
                 "1.2. Отдел выполняет:", "а. учет;", "б. отчетность."]:
        d.add_paragraph(line)
    d.save(path)
    doc = ingest_file(path, "after", "a1")
    assert ids(doc) == ["preamble", "1", "1.1", "1.2", "1.2.а", "1.2.б"]
    assert doc.clause("1.2.б").text == "отчетность."


def test_unnumbered_text_falls_back_to_paragraphs():
    clauses = parse_clauses(["Первый абзац.\n\nВторой абзац."], paged=False)
    assert [c.clause_id for c in clauses] == ["p1", "p2"]


def test_broken_file_gives_warning_not_crash(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    doc = ingest_file(path, "before", "b1")
    assert doc.clauses == [] and doc.warnings
