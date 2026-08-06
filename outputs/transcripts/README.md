# YouTube transcripts

Локальные транскрипты для конспектов в `topics/`. Имена файлов: `<VIDEO_ID>.txt`.

## Как скачать

```bash
uv sync --group tools
uv run python scripts/youtube_fetch_transcript.py "<YOUTUBE_URL_OR_ID>" \
  -o "outputs/transcripts/<VIDEO_ID>.txt" -v
uv run python scripts/verify_youtube_transcript.py "outputs/transcripts/<VIDEO_ID>.txt"
```

При блокировке YouTube (облачные IP):

```bash
uv run python scripts/youtube_fetch_transcript.py "<ID>" \
  --backend yt-dlp --cookies-from-browser chrome \
  -o "outputs/transcripts/<ID>.txt" -v
```

## Ожидаемые файлы из конспектов

| Video ID | Топик / конспект |
|----------|------------------|
| `C_GG5g38vLU` | [AI Harness Engineering (Tejas, IBM)](../../topics/code-agents-autoresearch-and-loopy-era/ai-harness-engineering-tejas-ibm.md) |

Если файла нет в репозитории — скачайте его локально командой выше, затем закоммитьте (`.gitignore` разрешает `outputs/transcripts/*.txt`).
