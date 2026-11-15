"""Собирает плоский ZIP только из .py и фиксированных разрешённых зависимостей."""
import argparse
import ast
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "config": "et_config",
    "llm.qwen": "et_llm",
    "consultant.contracts": "et_contracts",
    "consultant.dialogue": "et_dialogue",
    "consultant.answer_policy": "et_policy",
    "consultant.relations": "et_relations",
    "consultant.evidence": "et_evidence",
    "consultant.grounded": "et_grounded",
    "consultant.consultant": "et_consultant",
    "memory.conversation_state": "et_state",
    "prompts.system_prompts": "et_prompts",
    "rag.retrieval": "et_retrieval",
    "rag.retriever": "et_bm25",
    "rag.build_chunks": "et_chunks",
    "rag.embedder": "et_embedder",
    "rag.fusion": "et_fusion",
    "rag.query_normalization": "et_query",
    "rag.reranker": "et_reranker",
    "rag.ollama_reranker": "et_ollama_reranker",
}
REQUIREMENTS = "httpx==0.28.1\nnumpy==2.2.6\npython-dotenv==1.1.1\n"


class FlatImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module in MODULES:
            node.module = MODULES[node.module]
        elif (node.module or "").split(".")[0] in {"rag", "llm", "consultant", "memory", "prompts"}:
            raise ValueError("Неизвестный внутренний импорт: " + node.module)
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in MODULES:
                original = alias.name
                alias.name = MODULES[original]
                alias.asname = alias.asname or original
        return node


def flattened(source):
    tree = FlatImports().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for module, name in MODULES.items():
        source = ROOT / (module.replace(".", "/") + ".py")
        files[name + ".py"] = flattened(source.read_text(encoding="utf-8-sig"))
    files["solution.py"] = flattened((ROOT / "tools/platform_solution.py").read_text(encoding="utf-8"))
    files["requirements.txt"] = REQUIREMENTS
    whitelist = set((ROOT.parent.parent / "docs/reference/allowed_requirements.txt").read_text(encoding="utf-8-sig").splitlines())
    if not set(REQUIREMENTS.splitlines()).issubset(whitelist):
        raise ValueError("Зависимости не соответствуют выданному списку")
    for name, text in files.items():
        (output / name).write_text(text, encoding="utf-8")
    archive = output.parent / (output.name + ".zip")
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for name in files:
            z.writestr(name, files[name].encode("utf-8"))
    return {"archive": str(archive.resolve()), "files": len(files), "bytes": archive.stat().st_size}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "dist/platform")
    print(json.dumps(build(ap.parse_args().output), ensure_ascii=False))
