"""Run the pipeline from the terminal.

python -m backend.cli before1.pdf [before2.docx ...] --after after1.pdf [...] -o result.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

from backend.agent.pipeline import UnreadableInput, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнение комплектов документов «до» и «после»")
    parser.add_argument("before", nargs="+", help="файлы комплекта «до» (PDF/DOCX)")
    parser.add_argument("--after", nargs="+", required=True, help="файлы комплекта «после» (PDF/DOCX)")
    parser.add_argument("-o", "--output", default="result.json", help="куда записать результат (JSON)")
    parser.add_argument("--no-cache", action="store_true", help="не использовать кэш извлечения")
    args = parser.parse_args(argv)

    started = time.monotonic()

    def progress(step: str, fraction: float, message: str) -> None:
        print(f"[{time.monotonic() - started:6.1f}s {fraction:4.0%}] {step}: {message}", file=sys.stderr)

    try:
        result = run(args.before, args.after, progress=progress, use_cache=not args.no_cache)
    except UnreadableInput as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    Path(args.output).write_text(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    for step in result.trace:
        secs = (step.finished_at - step.started_at).total_seconds() if step.finished_at else 0
        tokens = f" in={step.input_tokens} out={step.output_tokens} cached={step.cached_tokens}" if step.model else ""
        cost = f" ${step.cost_usd}" if step.cost_usd is not None else ""
        print(f"  {step.step:<18} {secs:6.1f}s{tokens}{cost} | {step.notes}", file=sys.stderr)
    s = result.stats
    print(f"{args.output}: выводов {s.findings_total}, подтверждено {s.verified}, отклонено {s.rejected}; "
          f"полный анализ: {result.analysis_complete}; {s.duration_s} с; ${s.cost_usd}", file=sys.stderr)
    return 0 if result.analysis_complete else 2


if __name__ == "__main__":
    sys.exit(main())
