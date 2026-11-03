#!/usr/bin/env python3
"""
从南大云盘（Seafile）的**分享链接**下载课程数据包，并解压到指定目录。

用途
----
课程数据包（Stage A 预训练语料 + Stage B/C 武侠语料）放在南大云盘的分享链接里：

    https://box.nju.edu.cn/d/2b845e514e754429bb86/

浏览器里一个个点下载、再一个个 tar 解压，1.3 GB 的数据很折腾。本脚本一条命令做完：

  1. 调 `/api/v2.1/share-links/<token>/dirents/` **递归列举**分享内容
     （不需要账号、不需要 Cookie，分享链接自带匿名读权限）；
  2. 逐个下载到 `--dest`，**支持断点续传**（中断后重跑同一条命令即可接着下）；
  3. 下载完按 SHA-256 校验（清单见 `data/manifests/packed_data.sha256`）；
  4. 解压 `.tar` / `.tar.gz` / `.tar.bz2` / `.tar.xz` / `.zip` 到 `--extract-to`，
     非归档文件（如说明用的 `.md`）原样留下。

输入
----
* `--url`   分享链接（默认就是上面那条课程链接）
* `--dest`  归档文件落盘目录（默认 `<repo>/data/packed`，该目录已 gitignore）

输出
----
* `<dest>/*.tar*` 归档本体
* `<extract-to>/<顶层目录>/...` 解压结果（默认与 `--dest` 相同）
* 结束时打印一份"下了什么、多大、校验是否通过"的汇总

运行环境
--------
**纯标准库。不需要 torch，不需要 NPU，不需要账号。** 开发机与 NPU 服务器都能跑。
需要能访问 `box.nju.edu.cn`（校外访问需要走代理时，脚本自动识别
`http_proxy` / `https_proxy` 环境变量，和 curl 的行为一致）。

用法
----
    python scripts/fetch_box_data.py --list                # 先看看有什么
    python scripts/fetch_box_data.py                       # 全量下载 + 解压
    python scripts/fetch_box_data.py --only wuxia          # 只下 Stage C 的包
    python scripts/fetch_box_data.py --extract-to "$NANOCHAT_BASE_DIR"
    python scripts/fetch_box_data.py --no-extract --no-verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

HOST = "https://box.nju.edu.cn"
DEFAULT_URL = "https://box.nju.edu.cn/d/2b845e514e754429bb86/"
UA = "aics2026-course-fetcher/1.0"

# 归档后缀 → 处理方式。顺序敏感：先长后短（.tar.gz 要在 .gz 之前判）。
ARCHIVE_SUFFIXES = (
    ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar", ".zip",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SHA256_FILE = os.path.join(REPO_ROOT, "data", "manifests",
                                   "packed_data.sha256")


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def human(n: float) -> str:
    """把字节数格式化成人看的字符串。"""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def parse_token(url_or_token: str) -> str:
    """
    从分享链接里抠出 token。

    接受两种写法：
      https://box.nju.edu.cn/d/<token>/?p=%2F&mode=list   →  <token>
      <token>                                             →  <token>
    """
    s = url_or_token.strip()
    if "://" not in s:
        if not s or "/" in s:
            raise SystemExit(f"错误：看不懂的 token：{s!r}")
        return s
    parts = urllib.parse.urlsplit(s)
    segs = [p for p in parts.path.split("/") if p]
    # 形如 /d/<token>/ 或 /u/d/<token>/；token 紧跟在 d（或 u）之后。
    # 先找 d：/u/d/<token>/ 里的 u 后面跟的是 d 而不是 token。
    for tag in ("d", "u"):
        for i, seg in enumerate(segs):
            if seg == tag and i + 1 < len(segs):
                return segs[i + 1]
    raise SystemExit(f"错误：从链接里找不到 token：{s}")


def suffix_of(name: str) -> str:
    """返回归档后缀（小写），不是归档则返回空串。"""
    low = name.lower()
    for sfx in ARCHIVE_SUFFIXES:
        if low.endswith(sfx):
            return sfx
    return ""


# --------------------------------------------------------------------------
# Seafile 分享链接 API
# --------------------------------------------------------------------------
def api_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": UA,
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_dir(token: str, path: str, timeout: float = 30.0) -> dict:
    """
    列一个目录。
    """
    url = (f"{HOST}/api/v2.1/share-links/{token}/dirents/"
           f"?path={urllib.parse.quote(path)}")
    try:
        return json.loads(api_get(url, timeout))
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"错误：列举目录失败（HTTP {e.code}）：{path}\n"
            f"  {url}\n"
            f"  分享链接可能已过期、被取消，或需要登录。"
            f"请先用浏览器打开 {HOST}/d/{token}/ 确认还能访问。") from e


def walk(token: str, path: str = "/", depth: int = 0) -> list[dict]:
    """
    递归列举全部文件，返回 [{name, path, size}, ...]。
    """
    if depth > 16:
        raise SystemExit(f"错误：目录层级过深（>16），疑似链接成环：{path}")
    data = list_dir(token, path)
    files: list[dict] = []
    for ent in data.get("dirent_list", []):
        if ent.get("is_dir"):
            sub = ent.get("folder_path")
            if not sub:
                continue
            files.extend(walk(token, sub, depth + 1))
        else:
            fpath = ent.get("file_path")
            if not fpath:
                continue
            files.append({
                "name": ent.get("file_name") or os.path.basename(fpath.rstrip("/")),
                "path": fpath,
                "size": int(ent.get("size") or 0),
            })
    return files


def download_url(token: str, path: str) -> str:
    """拼直链。这个地址会 302 到 /seafhttp/files/<uuid>/<name>，用 urllib 自动跟随。"""
    return (f"{HOST}/d/{token}/files/"
            f"?p={urllib.parse.quote(path)}&dl=1")


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------
def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def read_expected_sha256(sha_file: str | None) -> dict[str, str]:
    """读 `sha256sum` 格式的清单：`<hash>  <filename>`。文件不存在就返回空表。"""
    out: dict[str, str] = {}
    if not sha_file or not os.path.isfile(sha_file):
        return out
    with open(sha_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[1].strip().lstrip("*")] = parts[0].strip().lower()
    return out


def download_one(token: str, item: dict, dest_dir: str,
                 retries: int = 4, quiet: bool = False) -> tuple[bool, str]:
    """
    下载单个文件，返回 (是否成功, 说明文本)。

    断点续传：先写 `<name>.part`，用 HTTP Range 从已有字节处接着下，
    下完再原子改名。中断后重跑同一条命令即可续传，不用从头再来。
    """
    name, size = item["name"], item["size"]
    final = os.path.join(dest_dir, name)
    part = final + ".part"

    if os.path.isfile(final) and (size <= 0 or os.path.getsize(final) == size):
        return True, f"已存在，跳过（{human(os.path.getsize(final))}）"

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            pos = os.path.getsize(part) if os.path.isfile(part) else 0
            if size and pos > size:      # 本地残留比远端还大，说明存坏了
                os.remove(part)
                pos = 0

            headers = {"User-Agent": UA, "Accept": "*/*"}
            if pos:
                headers["Range"] = f"bytes={pos}-"
            req = urllib.request.Request(
                download_url(token, item["path"]), headers=headers)

            with urllib.request.urlopen(req, timeout=120) as r:
                # 服务端不支持 Range 时会退化成 200，这时必须从头写
                resuming = pos > 0 and r.status == 206
                if pos and not resuming:
                    pos = 0
                mode = "ab" if resuming else "wb"
                got = pos
                t0 = time.monotonic()
                last_tick = 0.0
                with open(part, mode) as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        now = time.monotonic()
                        if not quiet and now - last_tick > 1.0:
                            last_tick = now
                            el = max(now - t0, 1e-6)
                            speed = (got - pos) / el
                            msg = (f"      {human(got)}"
                                   + (f"/{human(size)}" if size else "")
                                   + f"  {human(speed)}/s")
                            print("\r" + msg.ljust(60), end="", flush=True)
                if not quiet:
                    print("\r" + " " * 62 + "\r", end="")

            real = os.path.getsize(part)
            if size and real != size:
                raise IOError(f"体积不符：收到 {real} 字节，应为 {size} 字节")

            os.replace(part, final)
            return True, f"{human(real)}  完成"
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as e:
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode("utf-8", "replace")[:120]
                except Exception:
                    pass
            last_err = f"{type(e).__name__}: {e} {detail}".strip()
            if attempt < retries:
                wait = 3 * attempt
                if not quiet:
                    print(f"\n      ! 第 {attempt}/{retries} 次失败（{last_err}），"
                          f"{wait}s 后重试（已下 {human(os.path.getsize(part)) if os.path.isfile(part) else '0 B'}，"
                          f"将续传）", flush=True)
                time.sleep(wait)
    return False, f"失败（重试 {retries} 次）：{last_err}"


# --------------------------------------------------------------------------
# 解压
# --------------------------------------------------------------------------
def _common_top_level(names: list[str]) -> str | None:
    """
    如果归档里所有成员都在同一个顶层目录下，返回该目录名；否则 None。

    用途：先在临时目录解压、再整体改名，避免"解到一半"被当成解压完成。
    """
    tops = set()
    for n in names:
        n = n.replace("\\", "/").lstrip("./")
        if not n:
            continue
        tops.add(n.split("/", 1)[0])
        if len(tops) > 1:
            return None
    return tops.pop() if len(tops) == 1 else None


def _unsafe_member(name: str) -> bool:
    """拦掉绝对路径与 .. 穿越（tarfile/zipfile 都会中招）。"""
    n = name.replace("\\", "/")
    if n.startswith("/") or (len(n) > 1 and n[1] == ":"):
        return True
    return ".." in n.split("/")


def extract_one(archive: str, dest_dir: str, force: bool = False,
                quiet: bool = False) -> tuple[bool, str]:
    """解压一个归档到 dest_dir。返回 (是否成功, 说明)。"""
    name = os.path.basename(archive)
    if not suffix_of(name):
        return True, "非归档（如 .md 说明文件），跳过解压"

    try:
        if zipfile.is_zipfile(archive):
            members = zipfile.ZipFile(archive).namelist()
            top = _common_top_level(members)
            bad = [m for m in members if _unsafe_member(m)]
            if bad:
                return False, f"归档内含不安全路径，拒绝解压：{bad[:3]}"
            target = os.path.join(dest_dir, top) if top else None
            if target and os.path.isdir(target) and not force:
                return True, f"{top}/ 已存在，跳过解压（--force 可覆盖）"
            stage = os.path.join(dest_dir, f".stage_{name}")
            shutil.rmtree(stage, ignore_errors=True)
            os.makedirs(stage, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(stage)
        else:
            with tarfile.open(archive, "r:*") as tf:
                members = tf.getnames()
                bad = [m for m in members if _unsafe_member(m)]
                if bad:
                    return False, f"归档内含不安全路径，拒绝解压：{bad[:3]}"
                top = _common_top_level(members)
                target = os.path.join(dest_dir, top) if top else None
                if target and os.path.isdir(target) and not force:
                    return True, f"{top}/ 已存在，跳过解压（--force 可覆盖）"
                stage = os.path.join(dest_dir, f".stage_{name}")
                shutil.rmtree(stage, ignore_errors=True)
                os.makedirs(stage, exist_ok=True)
                try:
                    # Python ≥3.12 支持 filter="data"，会拦掉设备文件、
                    # 绝对路径与权限越界；老版本没有这个参数。
                    tf.extractall(stage, filter="data")
                except TypeError:
                    tf.extractall(stage)
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as e:
        shutil.rmtree(os.path.join(dest_dir, f".stage_{name}"),
                      ignore_errors=True)
        return False, f"解压失败：{type(e).__name__}: {e}"

    # 临时目录 → 正式目录（同一文件系统内的改名是原子的）
    moved: list[str] = []
    if top:
        src, dst = os.path.join(stage, top), os.path.join(dest_dir, top)
        if os.path.exists(dst):
            shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
        os.replace(src, dst)
        moved.append(top)
    else:
        for entry in os.listdir(stage):
            src, dst = os.path.join(stage, entry), os.path.join(dest_dir, entry)
            if os.path.exists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
            os.replace(src, dst)
            moved.append(entry)
    shutil.rmtree(stage, ignore_errors=True)
    if not quiet:
        print(f"      解压 → {', '.join(moved)}/")
    return True, f"解压 → {', '.join(moved)}/"


def check_stage_a(data_dir: str) -> None:
    """
    轻量校验 nanochat 的分片命名不变量：
    验证集必须**恰好一个**，且字典序排在所有训练分片之后。
    不满足的话多出来的 val 分片会被静默当成训练数据（会造成验证集泄漏）。
    """
    names = sorted(n for n in os.listdir(data_dir) if n.endswith(".parquet"))
    if not names:
        return
    train = [n for n in names if n.startswith("shard_train_")]
    val = [n for n in names if n.startswith("shard_val_")]
    if len(val) == 1 and names[-1] == val[0]:
        print(f"      ✓ 分片命名正确：{len(train)} 训练 + 1 验证，"
              f"验证集排在最后")
    else:
        print(f"      ! 分片命名可疑：{len(train)} train / {len(val)} val，"
              f"最后一个是 {names[-1]}")
        print("        验证集必须恰好 1 个且排在最后，否则会被当成训练数据。")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="从南大云盘分享链接下载并解压课程数据包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python scripts/fetch_box_data.py --list\n"
               "  python scripts/fetch_box_data.py --only wuxia\n"
               "  python scripts/fetch_box_data.py --extract-to $NANOCHAT_BASE_DIR\n")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"分享链接或 token（默认 {DEFAULT_URL}）")
    ap.add_argument("--dest", default=os.path.join(REPO_ROOT, "data", "packed"),
                    help="归档落盘目录（默认 <repo>/data/packed）")
    ap.add_argument("--extract-to", default=None,
                    help="解压目标目录（默认与 --dest 相同）")
    ap.add_argument("--only", action="append", default=[],
                    help="只处理名称含该子串的文件，可重复；"
                         "也可写 'a,b' 形式")
    ap.add_argument("--exclude", action="append", default=[],
                    help="跳过名称含该子串的文件，可重复")
    ap.add_argument("--list", action="store_true", help="只列出内容，不下载")
    ap.add_argument("--no-extract", action="store_true", help="只下载不解压")
    ap.add_argument("--force-extract", action="store_true",
                    help="目标目录已存在时也重新解压（会覆盖）")
    ap.add_argument("--delete-archives", action="store_true",
                    help="解压成功后删除归档，省磁盘（默认保留，便于续传与校验）")
    ap.add_argument("--sha256-file", default=DEFAULT_SHA256_FILE,
                    help=f"校验清单（默认 {os.path.relpath(DEFAULT_SHA256_FILE, REPO_ROOT)}，"
                         "不存在则跳过校验）")
    ap.add_argument("--no-verify", action="store_true", help="不做 SHA-256 校验")
    ap.add_argument("--retries", type=int, default=4, help="单文件重试次数")
    ap.add_argument("-q", "--quiet", action="store_true", help="安静模式")
    args = ap.parse_args()

    dest = os.path.abspath(os.path.expanduser(args.dest))
    extract_to = os.path.abspath(os.path.expanduser(
        args.extract_to or args.dest))

    token = parse_token(args.url)
    pats = [p.strip() for grp in args.only for p in grp.split(",") if p.strip()]
    skips = [p.strip() for grp in args.exclude for p in grp.split(",") if p.strip()]

    print(f"  分享链接: {HOST}/d/{token}/")
    print(f"  归档目录: {dest}")
    print(f"  解压目录: {extract_to}")
    if not args.no_verify:
        if os.path.isfile(args.sha256_file):
            print(f"  校验清单: {args.sha256_file}")
        else:
            print(f"  校验清单: 未找到（{args.sha256_file}），本轮不校验")
    print()

    # 1) 列举
    files = walk(token)
    if pats:
        files = [f for f in files if any(p in f["name"] for p in pats)]
    if skips:
        files = [f for f in files if not any(p in f["name"] for p in skips)]
    if not files:
        print("  分享链接里没有匹配的文件。用 --list 看看有什么。")
        return 1

    total = sum(f["size"] for f in files)
    print(f"  共 {len(files)} 个文件，合计 {human(total)}：")
    for f in files:
        kind = "归档" if suffix_of(f["name"]) else "普通"
        print(f"    {f['name']:<32} {human(f['size']):>10}  [{kind}]")
    print()

    if args.list:
        return 0

    os.makedirs(dest, exist_ok=True)
    if not args.no_extract:
        os.makedirs(extract_to, exist_ok=True)

    expected = {} if args.no_verify else read_expected_sha256(args.sha256_file)

    # 2) 下载
    print("  [下载]")
    t0 = time.monotonic()
    ok_files, bad_files, failed = [], [], []
    for i, item in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {item['name']}  ({human(item['size'])})")
        ok, msg = download_one(token, item, dest, retries=args.retries,
                               quiet=args.quiet)
        print(f"      {'✓' if ok else '✗'} {msg}")
        if not ok:
            failed.append(item["name"])
            continue
        ok_files.append(item)
        path = os.path.join(dest, item["name"])
        want = expected.get(item["name"])
        if want:
            got = sha256_of(path)
            if got == want:
                print(f"      ✓ SHA-256 一致 {got[:16]}…")
            else:
                print(f"      ✗ SHA-256 不符！")
                print(f"          期望 {want}")
                print(f"          实际 {got}")
                print("          文件可能被篡改或上传更新过；"
                      "确认无误后可用 --no-verify 跳过校验。")
                bad_files.append(item["name"])
        elif expected:
            print(f"      · 校验清单里没有 {item['name']} 的记录，跳过校验")

    dl_time = time.monotonic() - t0

    # 3) 解压
    extract_failed: list[str] = []
    if not args.no_extract:
        print("\n  [解压]")
        for i, item in enumerate(ok_files, 1):
            arc = os.path.join(dest, item["name"])
            print(f"  [{i}/{len(ok_files)}] {item['name']}")
            ok, msg = extract_one(arc, extract_to,
                                  force=args.force_extract, quiet=args.quiet)
            print(f"      {'✓' if ok else '✗'} {msg}")
            if not ok:
                extract_failed.append(item["name"])
                continue
            base = os.path.splitext(item["name"])[0]
            base = os.path.splitext(base)[0] if base.endswith(".tar") else base
            sub = os.path.join(extract_to, base)
            if os.path.isdir(sub) and base.startswith("base_data"):
                check_stage_a(sub)
            if args.delete_archives and suffix_of(item["name"]) and ok:
                os.remove(arc)
                print("      · 已删除归档（--delete-archives）")

    # 4) 汇总
    print("\n" + "=" * 70)
    print(f"  下载 {len(ok_files)}/{len(files)} 个文件，"
          f"{human(sum(f['size'] for f in ok_files))}，"
          f"耗时 {dl_time:.1f}s")
    if failed:
        print(f"  下载失败: {failed}")
    if bad_files:
        print(f"  校验失败: {bad_files}")
    if extract_failed:
        print(f"  解压失败: {extract_failed}")
    if failed or bad_files or extract_failed:
        print("  重跑同一条命令即可：已下好的会跳过，下了一半的会续传。")
    else:
        if not args.no_extract:
            print(f"  解压完成，产物在 {extract_to}")
            print("  若这是给训练用的，记得把 Stage A 数据放进 "
                  "$NANOCHAT_BASE_DIR，并跑：")
            print("    python scripts/check_budget.py --data-dir "
                  "$NANOCHAT_BASE_DIR/base_data_climbmix")
    print("=" * 70)
    return 1 if (failed or bad_files or extract_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
