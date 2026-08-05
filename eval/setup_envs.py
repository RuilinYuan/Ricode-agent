"""
环境搭建与验证
──────────────
对每个候选 instance：
  1. 下载 repo 源码 zip（codeload.github.com，带缓存）
  2. 解压到 eval/workspaces/<instance_id>/，git init 便于应用/回滚补丁
  3. 创建独立 venv（eval/venvs/<instance_id>），安装依赖
  4. 验证：应用 test_patch + gold patch → F2P 测试应全部通过
           回滚 gold patch        → F2P 测试应存在失败
  5. 验证通过且配额未满 → 加入正式任务集

输出：
  eval/data/valid_tasks.jsonl   正式任务（目标 30 个）
  eval/data/setup_report.json   每个候选的验证结果与失败原因
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

EVAL = Path(__file__).parent
DATA = EVAL / "data"
REPOS = EVAL / "repos"
WORKSPACES = EVAL / "workspaces"
VENVS = EVAL / "venvs"
for d in (REPOS, WORKSPACES, VENVS):
    d.mkdir(exist_ok=True)

QUOTA = {
    "django/django": 10,
    "sympy/sympy": 7,
    "pylint-dev/pylint": 6,
    "pytest-dev/pytest": 3,
    "sphinx-doc/sphinx": 2,
    "pallets/flask": 2,
}

# pip install -e . --no-deps 时需要手工补的运行时依赖
EXTRA_DEPS = {
    "django/django": ["asgiref", "sqlparse", "tzdata"],
    "sympy/sympy": ["mpmath"],
    "pylint-dev/pylint": ["astroid<3", "platformdirs", "dill", "isort", "mccabe", "toml", "tomli", "tomlkit", "typing-extensions"],
    "pytest-dev/pytest": ["iniconfig", "packaging", "pluggy", "tomli", "exceptiongroup", "pygments"],
    "sphinx-doc/sphinx": ["sphinxcontrib-applehelp", "sphinxcontrib-devhelp", "sphinxcontrib-htmlhelp",
                          "sphinxcontrib-jsmath", "sphinxcontrib-qthelp", "sphinxcontrib-serializinghtml",
                          "Jinja2", "Pygments", "docutils", "snowballstemmer", "babel", "alabaster",
                          "imagesize", "requests", "packaging", "importlib-metadata"],
    "pallets/flask": ["werkzeug<2.3", "jinja2", "click", "blinker", "itsdangerous", "importlib-metadata", "toml"],
}

# 老测试套件与新 pytest 不兼容时的 pytest 版本钉
PYTEST_PIN = {
    "pallets/flask": "pytest<8",
    "pylint-dev/pylint": "pytest<8",
}

# 安装时的环境变量覆盖（setuptools_scm 无 git 历史时需要假版本号）
INSTALL_ENV = {
    "pytest-dev/pytest": {"SETUPTOOLS_SCM_PRETEND_VERSION": "7.2.0"},
    "pallets/flask": {"SETUPTOOLS_SCM_PRETEND_VERSION": "2.2.0"},
    "pylint-dev/pylint": {"SETUPTOOLS_SCM_PRETEND_VERSION": "2.15.0"},
}

TEST_TIMEOUT = 600  # 单次测试运行上限（秒）


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], cwd: Path | None = None, timeout: int = TEST_TIMEOUT,
        env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, timeout=timeout,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)


def venv_python(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe"


def download_repo(repo: str, commit: str, dest_zip: Path) -> None:
    if dest_zip.exists():
        return
    url = f"https://codeload.github.com/{repo}/zip/{commit}"
    log(f"    下载 {url}")
    urllib.request.urlretrieve(url, dest_zip)


def _rmtree(path: Path) -> None:
    """Windows 下 git 对象/索引可能被锁或只读，chmod 后重试。"""
    import os
    import stat
    import time

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for _ in range(3):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except Exception:
            time.sleep(1)
    shutil.rmtree(path, ignore_errors=True)


def extract_workspace(instance_id: str, zip_path: Path) -> Path:
    ws = WORKSPACES / instance_id
    # marker 放在工作区外：git clean -fdq 会删掉工作区内的未跟踪文件
    marker_dir = WORKSPACES / ".extract_ok"
    marker_dir.mkdir(exist_ok=True)
    if (marker_dir / instance_id).exists():
        return ws
    if ws.exists():
        _rmtree(ws)
    ws.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ws)
    # zip 内含一层 <repo>-<commit>/ 目录，提升内容到 ws 根
    children = [c for c in ws.iterdir() if c.name != ".git"]
    if len(children) == 1 and children[0].is_dir():
        tmp = ws / "_tmp_move"
        children[0].rename(tmp)
        for item in tmp.iterdir():
            item.rename(ws / item.name)
        tmp.rmdir()
    # 健全性检查：提取结果必须非空（防中断残留的半拉子工作区）
    if not any(ws.iterdir()):
        raise RuntimeError("工作区提取为空")
    run(["git", "init", "-q"], cwd=ws)
    run(["git", "add", "-A"], cwd=ws)
    run(["git", "-c", "user.email=eval@local", "-c", "user.name=eval",
         "commit", "-qm", "base"], cwd=ws)
    (marker_dir / instance_id).touch()
    return ws


def create_venv(venv: Path) -> bool:
    if venv_python(venv).exists():
        return True
    r = run([sys.executable, "-m", "venv", str(venv)], timeout=300)
    if r.returncode != 0:
        log(f"    venv 创建失败: {r.stderr[-300:]}")
        return False
    run([str(venv_python(venv)), "-m", "pip", "install", "-q", "--upgrade", "pip"], timeout=300)
    return True


def install_deps(venv: Path, ws: Path, repo: str) -> bool:
    py = str(venv_python(venv))
    import os
    env = dict(os.environ)
    env.update(INSTALL_ENV.get(repo, {}))
    # 优先带依赖安装
    r = run([py, "-m", "pip", "install", "-q", "-e", "."], cwd=ws, timeout=900, env=env)
    if r.returncode != 0:
        log("    带依赖安装失败，改用 --no-deps + 手工依赖")
        r = run([py, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=ws, timeout=900, env=env)
        if r.returncode != 0:
            log("    可编辑安装也失败，改用 .pth 直接挂源码目录")
            # 老构建后端与新 setuptools 不兼容时：把仓库根目录写进 venv 的 .pth，
            # 包目录就在仓库根下（pylint/、src/ 布局则写 src）
            site = venv / "Lib" / "site-packages"
            src = ws / "src" if (ws / "src").is_dir() else ws
            (site / "_eval_workspace.pth").write_text(str(src) + "\n", encoding="utf-8")
        deps = EXTRA_DEPS.get(repo, [])
        if deps:
            r = run([py, "-m", "pip", "install", "-q"] + deps, timeout=900, env=env)
            if r.returncode != 0:
                log(f"    手工依赖安装失败: {(r.stderr or '')[-300:]}")
                return False
    # 即便带依赖安装成功，也要补齐仓库版本钉（如 flask 需要 werkzeug<2.3）
    pins = [d for d in EXTRA_DEPS.get(repo, []) if "<" in d or "==" in d]
    if pins:
        run([py, "-m", "pip", "install", "-q"] + pins, timeout=900, env=env)
    r = run([py, "-m", "pip", "install", "-q", PYTEST_PIN.get(repo, "pytest")], timeout=600, env=env)
    return r.returncode == 0


def f2p_to_labels(repo: str, f2p: list[str]) -> list[str]:
    """django 的 F2P 形如 'test_x (module.a.Class.test_x)'（混杂描述文本），
    转为 runtests.py 标签 'module.a.Class.test_x'；无法解析的条目丢弃。"""
    if repo != "django/django":
        return f2p
    import re
    labels = []
    for item in f2p:
        m = re.match(r"^(\w+)\s+\(([\w.]+)\)\s*$", item.strip())
        if not m:
            continue
        meth, path = m.groups()
        label = path if path.split(".")[-1] == meth else f"{path}.{meth}"
        if label.startswith("tests."):
            label = label[len("tests."):]
        labels.append(label)
    return labels


def _test_files_from_patch(patch_text: str) -> list[str]:
    """从 test_patch 的 diff 头里提取测试文件路径。"""
    import re
    files = []
    for m in re.finditer(r"^diff --git a/(\S+) b/\S+$", patch_text, re.M):
        path = m.group(1)
        if re.search(r"(^|/)(tests?|testing)/", path) and path.endswith(".py"):
            files.append(path)
    return files


def run_f2p(venv: Path, ws: Path, repo: str, f2p: list[str],
            test_patch: str = "") -> tuple[bool, str]:
    """运行 F2P 测试，返回 (全部通过, 输出尾部)。"""
    py = str(venv_python(venv))
    labels = f2p_to_labels(repo, f2p)
    if repo == "django/django":
        cmd = [py, "tests/runtests.py", "--verbosity", "1"] + labels
    elif all("/" not in x and "::" not in x for x in labels) and test_patch:
        # F2P 只有裸函数名（如 sympy）：从 test_patch 找测试文件，用 -k 过滤
        files = _test_files_from_patch(test_patch)
        if not files:
            return False, "test_patch 中找不到测试文件"
        k_expr = " or ".join(labels)
        cmd = [py, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider",
               "-k", k_expr] + files
    else:
        cmd = [py, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider"] + labels
    try:
        r = run(cmd, cwd=ws, timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    return r.returncode == 0, out[-500:]


def git_apply(ws: Path, patch_text: str) -> bool:
    p = ws / "_eval.patch"
    # newline="" 禁止 Windows 文本模式把 \n 转成 \r\n（git apply 对 CRLF 敏感）
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(patch_text.replace("\r\n", "\n"))
    r = run(["git", "apply", "--whitespace=nowarn", "_eval.patch"], cwd=ws)
    p.unlink(missing_ok=True)
    return r.returncode == 0


def git_reset(ws: Path) -> None:
    run(["git", "reset", "--hard", "-q"], cwd=ws)
    run(["git", "clean", "-fdq"], cwd=ws)


def validate_candidate(cand: dict) -> tuple[bool, str]:
    iid, repo = cand["instance_id"], cand["repo"]
    zip_path = REPOS / f"{iid}.zip"
    try:
        download_repo(repo, cand["base_commit"], zip_path)
        ws = extract_workspace(iid, zip_path)
        git_reset(ws)

        venv = VENVS / iid
        if not create_venv(venv):
            return False, "venv 创建失败"
        if not install_deps(venv, ws, repo):
            return False, "依赖安装失败"

        # 应用 test_patch（新增/修改测试文件）
        if not git_apply(ws, cand["test_patch"]):
            return False, "test_patch 应用失败"

        # 应用 gold patch → F2P 应全过
        if not git_apply(ws, cand["patch"]):
            git_reset(ws)
            return False, "gold patch 应用失败"
        ok, out = run_f2p(venv, ws, repo, cand["fail_to_pass"], cand["test_patch"])
        if not ok:
            git_reset(ws)
            return False, f"gold patch 下 F2P 未全过: {out[-200:]}"

        # 回滚 gold patch（保留 test_patch）→ F2P 应有失败
        git_reset(ws)
        git_apply(ws, cand["test_patch"])
        ok2, _ = run_f2p(venv, ws, repo, cand["fail_to_pass"], cand["test_patch"])
        git_reset(ws)
        if ok2:
            return False, "未打 gold patch 时 F2P 已全过（测试无鉴别力）"

        return True, "ok"
    except Exception as e:
        return False, f"异常: {type(e).__name__}: {str(e)[:200]}"


def main() -> None:
    cands = [json.loads(l) for l in (DATA / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    counts = {k: 0 for k in QUOTA}
    valid: list[dict] = []

    # 断点续跑：加载上次报告，已通过的直接计入
    report_path = DATA / "setup_report.json"
    report: dict[str, str] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    for cand in cands:
        repo = cand["repo"]
        if counts[repo] >= QUOTA[repo]:
            continue
        if report.get(cand["instance_id"]) == "ok":
            counts[repo] += 1
            valid.append(cand)
            log(f"[{cand['instance_id']}] 已验证过，跳过")
            continue
        log(f"[{cand['instance_id']}] 验证中…")
        ok, reason = validate_candidate(cand)
        report[cand["instance_id"]] = reason
        if ok:
            counts[repo] += 1
            valid.append(cand)
            log(f"    [OK] 通过（{repo} {counts[repo]}/{QUOTA[repo]}）")
        else:
            log(f"    [X] 淘汰：{reason}")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 回填：总数不足 30 时，从剩余候选（无视配额）补足 ─────────────────
    TARGET = 30
    if len(valid) < TARGET:
        valid_ids = {v["instance_id"] for v in valid}
        for cand in cands:
            if len(valid) >= TARGET:
                break
            iid = cand["instance_id"]
            if iid in valid_ids or report.get(iid) == "ok":
                continue
            log(f"[{iid}] 回填验证中…")
            ok, reason = validate_candidate(cand)
            report[iid] = reason
            if ok:
                valid.append(cand)
                valid_ids.add(iid)
                log(f"    [OK] 回填通过（{len(valid)}/{TARGET}）")
            else:
                log(f"    [X] 淘汰：{reason}")
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (DATA / "valid_tasks.jsonl").open("w", encoding="utf-8") as f:
        for v in valid:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    log(f"\n有效任务 {len(valid)} 个 → data/valid_tasks.jsonl")
    log(str(counts))


if __name__ == "__main__":
    main()
