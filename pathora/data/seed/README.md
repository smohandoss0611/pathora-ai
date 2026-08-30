# Sample data

All students, schools and statistics here are **fictional**.

| File | What it is |
|---|---|
| `sample_transcript.txt` | Strong STEM transcript — 5 AP courses, calculus, two CS courses. Computed GPA 3.73. Parses cleanly, so the workflow runs without stopping. |
| `sample_transcript.pdf` | The same transcript as a PDF, for testing the PyMuPDF path. |
| `sample_transcript_weaker.txt` | Contrasting profile — no AP, no calculus, GPA 3.09, no class rank. Triggers the human-verification interrupt and produces a very different college list. |
| `sample_student_profile.json` | Test scores, activities, projects, awards and preferences. Maps to `student_input` on `POST /analysis/start`. |
| `colleges.json` | Ten fictional universities and their official-document corpus. |

## Using them

**Streamlit** — upload a transcript file; enter the profile in the sidebar
(SAT 1450, interests Computer Science + Data Science, state TX, Public, Large).

**API** —

```bash
python - <<'PY' > /tmp/req.json
import json, pathlib
profile = {k: v for k, v in
           json.loads(pathlib.Path("data/seed/sample_student_profile.json").read_text()).items()
           if not k.startswith("_")}
print(json.dumps({
    "thread_id": "sample-1",
    "user_id": "demo-user",
    "student_id": "demo-student",
    "transcript_text": pathlib.Path("data/seed/sample_transcript.txt").read_text(),
    "student_input": profile,
}))
PY

curl -s -X POST localhost:8000/analysis/start -H 'content-type: application/json' -d @/tmp/req.json | jq '.workflow_status, .state.admission_results | keys'
```

The `_note` key exists for humans; strip keys beginning with `_` before sending,
as the models use `extra="forbid"`.

**Terminal, no setup** — `python scripts/demo.py` runs the strong profile end to
end and prints every stage.
