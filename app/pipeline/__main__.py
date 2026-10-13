"""Run with python -m app.pipeline PDF --output OUTPUT.json."""
import argparse
import json
from pathlib import Path
import sys
from . import process_pdf, ChunkConfig, EmbeddingService
from .ingest import IngestionError
from .embed import EmbeddingError


def main() -> int:
    parser = argparse.ArgumentParser(description='Extract, section-chunk, and embed a research PDF')
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-tokens', type=int, default=220)
    parser.add_argument('--overlap-tokens', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--revision')
    parser.add_argument('--no-heading-prefix', action='store_true')
    parser.add_argument('--exclude-references', action='store_true')
    parser.add_argument('--exclude-appendix', action='store_true')
    args = parser.parse_args()
    try:
        if args.pdf.resolve() == args.output.resolve():
            raise ValueError('Output must differ from the input PDF path.')
        result = process_pdf(args.pdf, config=ChunkConfig(args.target_tokens, args.overlap_tokens,
                             not args.no_heading_prefix, args.exclude_references, args.exclude_appendix),
                             service=EmbeddingService(batch_size=args.batch_size, device=args.device, revision=args.revision))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    except (OSError, IngestionError, EmbeddingError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    print(f"Wrote {len(result['chunks'])} chunks to {args.output}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
