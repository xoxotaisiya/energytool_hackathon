"""Контракт реального плоского ZIP в отдельном процессе с управляемым API."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from tools.build_submission import build

ROOT = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def test_flat_export_runs_without_source_packages_and_keeps_cases_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            flat = work / "flat"
            built = build(flat)
            with ZipFile(built["archive"]) as z:
                names = z.namelist()
                self.assertTrue(all("/" not in n and "\\" not in n for n in names))
                self.assertTrue(all(n.endswith(".py") or n == "requirements.txt" for n in names))
                self.assertIn("solution.py", names)
                self.assertLess(built["bytes"], 128*1024*1024)
            data = work / "data"
            data.mkdir()
            with ZipFile(data / "knowledge-base.zip", "w") as z:
                for p in (ROOT / "knowledge-base").rglob("*.md"):
                    z.write(p, p.relative_to(ROOT / "knowledge-base").as_posix())
            with (data / "questions.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "question"])
                writer.writerows([
                    ["001", "CLIENT_A_ONLY. У нас крыша на два ската — восток и запад. Панели можно поставить на оба?"],
                    ["002", "Пользователь: У нас нет места на крыше.\nАссистент: Рассмотрим размещение панелей.\nПользователь: На фасаде их можно поставить?"],
                    ["003", "Здравствуйте"],
                ])
            script = r'''
import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import httpx
original_client = httpx.Client
trace = []
def handler(request):
    body=json.loads(request.content)
    if request.url.path == '/v1/embeddings':
        assert body['model']=='baai/bge-m3'
        if any('CLIENT_A_ONLY' in text for text in body['input']):
            return httpx.Response(503)
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.,0.]} for i in range(len(body['input']))]})
    assert request.url.path == '/v1/chat/completions'
    assert body['model']=='test-platform-Qwen'
    assert 'format' not in body and 'think' not in body
    trace.append(body['model'])
    if 'response_format' not in body:
        content='готов'
    else:
        assert body['response_format']=={'type':'json_object'}
        data=json.loads(body['messages'][1]['content'])
        if 'фасаде' in data['history']:
            assert 'CLIENT_A_ONLY' not in data['history']
            assert 'roof_orientation' not in data['customer_facts']
        if 'выбираешь подтверждения' in body['messages'][0]['content']:
            selected=next(u for u in data['evidence'] if ('Ориентация скатов' in u['quote'] if 'CLIENT_A_ONLY' in data['history'] else 'вертикальные поверхности' in u['quote']))
            value={'intent':'product_question','out_of_scope':False,'wants_human':False,'evidence_ids':[selected['id']]}
        else:
            fact=data['evidence'][0]
            value={'action':'answered','claims':[{'text':fact['quote'],'evidence_ids':[fact['id']]}]}
        content=json.dumps(value,ensure_ascii=False)
    return httpx.Response(200,json={'choices':[{'message':{'content':content},'finish_reason':'stop'}]})
def client(*args,**kwargs):
    kwargs['transport']=httpx.MockTransport(handler)
    return original_client(*args,**kwargs)
httpx.Client=client
import solution
assert solution.main()==0
Path('api_trace.json').write_text(json.dumps(trace),encoding='utf-8')
'''
            env = os.environ.copy()
            for key in ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "GEN_MODEL", "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "KB_PATH"):
                env.pop(key, None)
            env.update({"OPENAI_BASE_URL": "https://platform.invalid/v1", "OPENAI_API_KEY": "test-placeholder",
                        "GEN_MODEL": "test-platform-Qwen", "DATA_DIR": str(data), "MAX_WORKERS": "2",
                        "RETRIEVER_MODE": "dense", "EMB_MODEL": "baai/bge-m3", "GROUNDED_ANSWERS": "1", "RERANKER_ENABLED": "0"})
            result = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(flat)], cwd=work, env=env,
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("test-placeholder", result.stderr+result.stdout)
            self.assertIn("prefetch_fallback=BM25", result.stderr)
            with (work / "submission.csv").open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["id", "answer"])
                rows = list(reader)
            self.assertEqual([r["id"] for r in rows], ["001", "002", "003"])
            self.assertIn("оба ската", rows[0]["answer"])
            self.assertIn("вертикальные поверхности", rows[1]["answer"])
            self.assertTrue(all(r["answer"] and len(r["answer"]) <= 10000 for r in rows))
            self.assertEqual(len(json.loads((work / "api_trace.json").read_text())), 5)


if __name__ == "__main__":
    unittest.main()
