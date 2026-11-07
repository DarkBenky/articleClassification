"""Usage:
    python shuffle.py corpus.txt              # shuffle in place
    python shuffle.py corpus.txt out.txt      # write a shuffled copy
    python shuffle.py corpus.txt --seed 42    # reproducible order
"""

import argparse
import hashlib
import mmap
import os
import secrets
import sys
import tempfile
import time
from array import array

import numpy as np

MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1
BUFFER_SIZE = 1 << 20


def lineDigest(line):
    digest = hashlib.blake2b(line.rstrip(b"\r\n"), digest_size=16).digest()
    return int.from_bytes(digest, "little")


def scan(path):
    offsets = array("Q", [0])
    digestSum = 0
    digestXor = 0
    payloadBytes = 0
    totalBytes = 0
    with open(path, "rb") as f:
        for line in f:
            totalBytes += len(line)
            offsets.append(totalBytes)
            digest = lineDigest(line)
            digestSum = (digestSum + digest) & MASK128
            digestXor ^= digest
            payloadBytes += len(line.rstrip(b"\r\n"))
    return {
        "offsets": offsets,
        "lines": len(offsets) - 1,
        "totalBytes": totalBytes,
        "payloadBytes": payloadBytes,
        "digestSum": digestSum,
        "digestXor": digestXor,
    }


def writeShuffled(src, dst, offsets, seed):
    count = len(offsets) - 1
    order = np.random.default_rng(seed).permutation(count) if count > 1 else range(count)
    addedNewlines = 0
    with open(dst, "wb", buffering=BUFFER_SIZE) as out, open(src, "rb") as f:
        if count == 0:
            return addedNewlines
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for index in order:
                line = mm[offsets[index]:offsets[index + 1]]
                if not line.endswith(b"\n"):
                    line += b"\n"
                    addedNewlines += 1
                out.write(line)
    return addedNewlines


def sameContent(before, after):
    return (
        before["lines"] == after["lines"]
        and before["payloadBytes"] == after["payloadBytes"]
        and before["digestSum"] == after["digestSum"]
        and before["digestXor"] == after["digestXor"]
    )


def formatBytes(count):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if count < 1024 or unit == "TiB":
            return f"{count:,} B" if unit == "B" else f"{count:,.1f} {unit}"
        count /= 1024


def shuffleFile(src, dst=None, seed=None):
    src = os.path.realpath(src)
    dst = os.path.realpath(dst or src)
    dstDir = os.path.dirname(dst) or "."
    os.makedirs(dstDir, exist_ok=True)

    try:
        sourceStat = os.stat(src)
    except FileNotFoundError:
        raise SystemExit(f"error: no such file: {src}")
    if not os.path.isfile(src):
        raise SystemExit(f"error: not a regular file: {src}")

    if seed is None:
        seed = secrets.randbits(64)

    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=dstDir)
    os.close(fd)
    started = time.monotonic()
    try:
        before = scan(src)
        writeShuffled(src, tmp, before["offsets"], seed)
        after = scan(tmp)

        print(f"file:        {dst}")
        print(f"size:        {formatBytes(before['totalBytes'])}")
        print(f"lines:       {before['lines']:,} -> {after['lines']:,}")
        print(f"payload:     {before['payloadBytes']:,} -> {after['payloadBytes']:,} bytes")
        print(f"fingerprint: {before['digestSum'] & MASK64:016x} -> {after['digestSum'] & MASK64:016x}")
        print(f"xor print:   {before['digestXor'] & MASK64:016x} -> {after['digestXor'] & MASK64:016x}")
        print(f"seed:        {seed}")

        if not sameContent(before, after):
            raise RuntimeError("verification failed, source left untouched")
        if after["totalBytes"] != before["totalBytes"]:
            print("note:        added a trailing newline to the last line")
        if before["lines"] <= 1:
            print("note:        one line or less, order is trivially unchanged")

        os.chmod(tmp, sourceStat.st_mode & 0o7777)
        try:
            os.chown(tmp, sourceStat.st_uid, sourceStat.st_gid)
        except (PermissionError, AttributeError):
            pass

        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, dst)
        try:
            dirFd = os.open(dstDir, os.O_RDONLY)
        except OSError:
            dirFd = None
        if dirFd is not None:
            try:
                os.fsync(dirFd)
            finally:
                os.close(dirFd)

        elapsed = max(time.monotonic() - started, 1e-9)
        rate = formatBytes(int(before["totalBytes"] / elapsed))
        print(f"OK in {elapsed:.1f} s ({rate}/s)")
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle the lines of a text file (in place by default)."
    )
    parser.add_argument("src", help="file to shuffle")
    parser.add_argument(
        "dst", nargs="?", default=None,
        help="output file (default: shuffle the source in place)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for a reproducible order (default: random)",
    )
    args = parser.parse_args()
    shuffleFile(args.src, args.dst, args.seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted, source left untouched", file=sys.stderr)
        sys.exit(130)