#!/usr/bin/env python3
"""Prepare the real-data and controlled join workloads.

Subcommands:
  job1a         Project the five-relation acyclic join core of JOB/IMDb Q1a.
  job13d        Project the nine-relation acyclic join core of JOB/IMDb Q13d.
  job13d-scales Build 20%, 40%, 60%, 80%, and 100% JOB/IMDb inputs.
  snap-acyclic  Build controlled acyclic joins from SNAP email-EuAll.
  tpch9         Generate and project the six-relation TPC-H Q9 join tree.

The SNAP suite uses only real email edges. It contains medium- and large-scale
controlled series for join-tree depth and branching factor, plus a relation-
count series. Each case is stored in tree.txt, tables.tbl, expected.txt, and
stats.txt.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


JOB_URL = "https://event.cwi.nl/da/job/imdb.tgz"
SNAP_EMAIL_URL = "https://snap.stanford.edu/data/email-EuAll.txt.gz"
INT_MAX = 2_147_483_647


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"[reuse] {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
        print(f"[resume] {url} from byte {offset}", flush=True)
    else:
        print(f"[download] {url}", flush=True)
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    partial.replace(destination)
    print(f"[downloaded] {destination}", flush=True)


def require_input(path: Path, url: str, allow_download: bool) -> Path:
    if not path.exists() and allow_download:
        download(url, path)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; pass --download or download it from {url}"
        )
    return path


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is not empty: {path}; use --force to replace generated files"
        )
    path.mkdir(parents=True, exist_ok=True)


def write_int_row(out: TextIO, values: list[int] | tuple[int, ...]) -> None:
    out.write(" ".join(str(value) for value in values))
    out.write("\n")


TPCH_TABLES = (
    "orders",
    "lineitem",
    "partsupp",
    "part",
    "supplier",
    "nation",
)


def tpch_rows(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        for row in csv.reader(stream, delimiter="|"):
            if row and row[-1] == "":
                row.pop()
            if row:
                yield row


def ensure_tpch_data(dbgen_dir: Path, raw_dir: Path, scale: str) -> None:
    if all((raw_dir / f"{table}.tbl").is_file() for table in TPCH_TABLES):
        print(f"[reuse] TPC-H raw data: {raw_dir}", flush=True)
        return
    if sys.platform.startswith("win"):
        raise OSError("TPC-H dbgen preparation must run on Linux")

    dbgen = dbgen_dir / "dbgen"
    distributions = dbgen_dir / "dists.dss"
    if not distributions.is_file():
        raise FileNotFoundError(f"missing {distributions}")
    data = distributions.read_bytes()
    normalized = data.replace(b"\r\n", b"\n")
    if normalized != data:
        distributions.write_bytes(normalized)

    if not dbgen.is_file():
        for path in dbgen_dir.glob("*.o"):
            path.unlink(missing_ok=True)
        for path in dbgen_dir.glob("*.exe"):
            path.unlink(missing_ok=True)
        subprocess.run(["make", "dbgen"], cwd=str(dbgen_dir), check=True)
    if not dbgen.is_file():
        raise FileNotFoundError(f"failed to build {dbgen}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate] TPC-H scale={scale}: {raw_dir}", flush=True)
    subprocess.run(
        [str(dbgen), "-f", "-s", scale, "-b", str(distributions)],
        cwd=str(raw_dir),
        check=True,
    )
    missing = [table for table in TPCH_TABLES if not (raw_dir / f"{table}.tbl").is_file()]
    if missing:
        raise FileNotFoundError(f"TPC-H dbgen did not create: {', '.join(missing)}")


def export_tpch9(
    dbgen_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    scale: str,
    force: bool,
    green_only: bool,
) -> None:
    """Project the Q9 join before its final aggregation.

    The full workload evaluates the Q9 join tree before the final aggregation
    and therefore keeps all part keys.  ``green_only`` is retained
    only as an optional smoke-test variant for the SQL predicate
    p_name LIKE '%green%'. Each projected relation keeps only the integer
    columns consumed by App.cpp.
    """
    ensure_tpch_data(dbgen_dir, raw_dir, scale)
    prepare_output_dir(out_dir, force)

    selected_parts: set[int] = set()
    with (out_dir / "part.txt").open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(tpch_rows(raw_dir / "part.tbl"), start=1):
            require_columns(row, 2, "part", row_no)
            part_key = as_int(row[0], "part", row_no, 0)
            if not green_only or "green" in row[1]:
                selected_parts.add(part_key)
                write_int_row(out, [part_key])

    order_keys: set[int] = set()
    with (out_dir / "orders_ok.txt").open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(tpch_rows(raw_dir / "orders.tbl"), start=1):
            require_columns(row, 1, "orders", row_no)
            order_key = as_int(row[0], "orders", row_no, 0)
            order_keys.add(order_key)
            write_int_row(out, [order_key])

    nation_keys: set[int] = set()
    with (out_dir / "nation.txt").open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(tpch_rows(raw_dir / "nation.tbl"), start=1):
            require_columns(row, 1, "nation", row_no)
            nation_key = as_int(row[0], "nation", row_no, 0)
            nation_keys.add(nation_key)
            write_int_row(out, [nation_key])

    supplier_keys: set[int] = set()
    with (out_dir / "supplier_sk_nk.txt").open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(tpch_rows(raw_dir / "supplier.tbl"), start=1):
            require_columns(row, 4, "supplier", row_no)
            supplier_key = as_int(row[0], "supplier", row_no, 0)
            nation_key = as_int(row[3], "supplier", row_no, 3)
            if nation_key in nation_keys:
                supplier_keys.add(supplier_key)
                write_int_row(out, [supplier_key, nation_key])

    part_supplier_keys: set[tuple[int, int]] = set()
    with (out_dir / "partsupp.txt").open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(tpch_rows(raw_dir / "partsupp.tbl"), start=1):
            require_columns(row, 2, "partsupp", row_no)
            part_key = as_int(row[0], "partsupp", row_no, 0)
            supplier_key = as_int(row[1], "partsupp", row_no, 1)
            part_supplier_keys.add((part_key, supplier_key))
            write_int_row(out, [supplier_key, part_key])

    lineitem_rows = 0
    expected = 0
    with (out_dir / "lineitem_sk_pk_ok.txt").open("w", encoding="utf-8") as out:
        for row_no, row in enumerate(tpch_rows(raw_dir / "lineitem.tbl"), start=1):
            require_columns(row, 3, "lineitem", row_no)
            order_key = as_int(row[0], "lineitem", row_no, 0)
            part_key = as_int(row[1], "lineitem", row_no, 1)
            supplier_key = as_int(row[2], "lineitem", row_no, 2)
            write_int_row(out, [supplier_key, part_key, order_key])
            lineitem_rows += 1
            if (
                part_key in selected_parts
                and order_key in order_keys
                and supplier_key in supplier_keys
                and (part_key, supplier_key) in part_supplier_keys
            ):
                expected += 1

    if not selected_parts or expected <= 0 or expected > INT_MAX:
        raise ValueError(f"TPC-H Q9 output {expected} is outside the benchmark limit")
    counts = {
        "orders": len(order_keys),
        "lineitem": lineitem_rows,
        "partsupp": len(part_supplier_keys),
        "part_selected": len(selected_parts),
        "supplier": len(supplier_keys),
        "nation": len(nation_keys),
    }
    total_input = sum(counts.values())
    (out_dir / "expected.txt").write_text(f"{expected}\n", encoding="utf-8")
    with (out_dir / "stats.txt").open("w", encoding="utf-8") as out:
        out.write("workload=TPC-H Q9 join before final aggregation\n")
        out.write(f"scale={scale}\n")
        out.write(f"part_filter={'green' if green_only else 'none'}\n")
        out.write("relations=6\n")
        out.write("depth=3\n")
        out.write("branching=2\n")
        out.write(f"input_rows={total_input}\n")
        out.write(f"output_rows={expected}\n")
        for name, count in counts.items():
            out.write(f"{name}={count}\n")
    print(
        f"[prepared] TPC-H Q9: {out_dir} input={total_input} output={expected}",
        flush=True,
    )


class JobCsvSource:
    """Read JOB CSV tables either from a directory or directly from imdb.tgz."""

    def __init__(self, source: Path):
        self.source = source
        self.archive: tarfile.TarFile | None = None
        self.members: dict[str, tarfile.TarInfo] = {}

    def __enter__(self) -> "JobCsvSource":
        if self.source.is_file():
            self.archive = tarfile.open(self.source, "r:*")
            for member in self.archive.getmembers():
                if member.isfile():
                    self.members[Path(member.name).name] = member
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.archive is not None:
            self.archive.close()

    @contextmanager
    def open_table(self, table: str) -> Iterator[TextIO]:
        names = (f"{table}.csv", f"{table}.csv.gz")
        if self.source.is_dir():
            path = next(
                (self.source / name for name in names if (self.source / name).exists()),
                None,
            )
            if path is None:
                matches = [
                    candidate for name in names for candidate in self.source.rglob(name)
                ]
                path = matches[0] if matches else None
            if path is None:
                raise FileNotFoundError(f"missing {table}.csv under {self.source}")
            if path.suffix == ".gz":
                with gzip.open(
                    path, "rt", encoding="utf-8", errors="replace", newline=""
                ) as stream:
                    yield stream
            else:
                with path.open(
                    "r", encoding="utf-8", errors="replace", newline=""
                ) as stream:
                    yield stream
            return

        if self.archive is None:
            raise RuntimeError("JOB archive is not open")
        member = next((self.members[name] for name in names if name in self.members), None)
        if member is None:
            raise FileNotFoundError(f"missing {table}.csv in {self.source}")
        raw = self.archive.extractfile(member)
        if raw is None:
            raise OSError(f"cannot read {member.name} from {self.source}")
        with raw:
            binary = gzip.GzipFile(fileobj=raw) if member.name.endswith(".gz") else raw
            with io.TextIOWrapper(
                binary, encoding="utf-8", errors="replace", newline=""
            ) as stream:
                yield stream

    def rows(self, table: str) -> Iterator[list[str]]:
        with self.open_table(table) as stream:
            # IMDb text fields use backslash-escaped quotes.
            for row in csv.reader(
                stream, delimiter=",", quotechar='"', escapechar="\\"
            ):
                if row:
                    yield row


def require_columns(row: list[str], count: int, table: str, row_no: int) -> None:
    if len(row) < count:
        raise ValueError(
            f"{table} row {row_no} has {len(row)} columns; expected at least {count}"
        )


def as_int(value: str, table: str, row_no: int, column: int) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{table} row {row_no} column {column} is not an integer: {value!r}"
        ) from exc
    if result < -INT_MAX - 1 or result > INT_MAX:
        raise ValueError(f"{table} row {row_no} has an integer outside int32")
    return result


def export_job1a(source_path: Path, out_dir: Path, force: bool) -> None:
    prepare_output_dir(out_dir, force)
    counts: Counter[str] = Counter()

    with JobCsvSource(source_path) as source:
        company_type_ids: set[int] = set()
        with (out_dir / "R2_company_type.tbl").open("w", encoding="utf-8") as out:
            for row_no, row in enumerate(source.rows("company_type"), start=1):
                require_columns(row, 2, "company_type", row_no)
                if row[1] == "production companies":
                    key = as_int(row[0], "company_type", row_no, 0)
                    company_type_ids.add(key)
                    write_int_row(out, [key])
                    counts["R2_company_type"] += 1

        info_type_ids: set[int] = set()
        with (out_dir / "R5_info_type.tbl").open("w", encoding="utf-8") as out:
            for row_no, row in enumerate(source.rows("info_type"), start=1):
                require_columns(row, 2, "info_type", row_no)
                if row[1] == "top 250 rank":
                    key = as_int(row[0], "info_type", row_no, 0)
                    info_type_ids.add(key)
                    write_int_row(out, [key])
                    counts["R5_info_type"] += 1

        if not company_type_ids or not info_type_ids:
            raise ValueError(
                "JOB 1a predicate values were not found; use the original JOB IMDb snapshot"
            )

        title_ids: set[int] = set()
        with (out_dir / "R3_title.tbl").open("w", encoding="utf-8") as out:
            for row_no, row in enumerate(source.rows("title"), start=1):
                require_columns(row, 1, "title", row_no)
                movie_id = as_int(row[0], "title", row_no, 0)
                title_ids.add(movie_id)
                write_int_row(out, [movie_id])
                counts["R3_title"] += 1

        selected_info_by_movie: Counter[int] = Counter()
        with (out_dir / "R4_movie_info_idx.tbl").open("w", encoding="utf-8") as out:
            for row_no, row in enumerate(source.rows("movie_info_idx"), start=1):
                require_columns(row, 3, "movie_info_idx", row_no)
                movie_id = as_int(row[1], "movie_info_idx", row_no, 1)
                info_type_id = as_int(row[2], "movie_info_idx", row_no, 2)
                write_int_row(out, [movie_id, info_type_id])
                counts["R4_movie_info_idx"] += 1
                if movie_id in title_ids and info_type_id in info_type_ids:
                    selected_info_by_movie[movie_id] += 1

        expected = 0
        with (out_dir / "R1_movie_companies.tbl").open("w", encoding="utf-8") as out:
            for row_no, row in enumerate(source.rows("movie_companies"), start=1):
                require_columns(row, 5, "movie_companies", row_no)
                note = row[4]
                selected = (
                    "(as Metro-Goldwyn-Mayer Pictures)" not in note
                    and ("(co-production)" in note or "(presents)" in note)
                )
                if not selected:
                    continue
                movie_id = as_int(row[1], "movie_companies", row_no, 1)
                company_type_id = as_int(row[3], "movie_companies", row_no, 3)
                write_int_row(out, [company_type_id, movie_id])
                counts["R1_movie_companies"] += 1
                if company_type_id in company_type_ids and movie_id in title_ids:
                    expected += selected_info_by_movie[movie_id]

    if counts["R1_movie_companies"] == 0 or expected == 0:
        raise ValueError(
            "JOB 1a produced an empty projected join; use the May 2013 JOB IMDb snapshot"
        )
    if expected > INT_MAX:
        raise ValueError(f"JOB 1a output {expected} exceeds the benchmark int row limit")

    relation_names = (
        "R1_movie_companies",
        "R2_company_type",
        "R3_title",
        "R4_movie_info_idx",
        "R5_info_type",
    )
    total_input = sum(counts[name] for name in relation_names)
    (out_dir / "expected.txt").write_text(f"{expected}\n", encoding="utf-8")
    with (out_dir / "stats.txt").open("w", encoding="utf-8") as out:
        out.write("workload=JOB-1a join core before MIN aggregation\n")
        out.write("query_url=https://github.com/gregrahn/join-order-benchmark/blob/master/1a.sql\n")
        out.write(f"source={source_path.name}\n")
        out.write("relations=5\n")
        out.write("depth=2\n")
        out.write("branching=3\n")
        out.write(f"input_rows={total_input}\n")
        out.write(f"output_rows={expected}\n")
        for name in relation_names:
            out.write(f"{name}={counts[name]}\n")
    print(
        f"[prepared] JOB 1a: {out_dir} input={total_input} output={expected}",
        flush=True,
    )


def job13d_sample_bucket(movie_id: int) -> int:
    """Map a title id to a stable bucket for nested percentage samples."""
    return ((movie_id * 2_654_435_761) & 0xFFFF_FFFF) % 100


def export_job13d(
    source_path: Path,
    out_dir: Path,
    force: bool,
    sample_percent: int = 100,
) -> None:
    """Project the join before Q13d's final MIN aggregation.

    The tree is rooted at title.  Its four branches are kind_type,
    movie_companies, movie_info, and movie_info_idx; the last three have their
    filtered dimension relations below them.  Redundant fact-to-fact movie_id
    equalities in the SQL follow transitively through title.
    """
    if sample_percent < 1 or sample_percent > 100:
        raise ValueError("JOB 13d sample percentage must be in [1, 100]")
    prepare_output_dir(out_dir, force)
    counts: Counter[str] = Counter()
    tables_path = out_dir / "tables.tbl"

    with JobCsvSource(source_path) as source, tables_path.open(
        "w", encoding="utf-8"
    ) as out:
        production_company_type_ids: set[int] = set()
        for row_no, row in enumerate(source.rows("company_type"), start=1):
            require_columns(row, 2, "company_type", row_no)
            if row[1] == "production companies":
                key = as_int(row[0], "company_type", row_no, 0)
                production_company_type_ids.add(key)
                write_int_row(out, [4, key])
                counts["R4_company_type"] += 1

        info_type_ids: dict[str, set[int]] = {
            "release dates": set(),
            "rating": set(),
        }
        for row_no, row in enumerate(source.rows("info_type"), start=1):
            require_columns(row, 2, "info_type", row_no)
            if row[1] in info_type_ids:
                key = as_int(row[0], "info_type", row_no, 0)
                info_type_ids[row[1]].add(key)
                relation = 6 if row[1] == "release dates" else 8
                write_int_row(out, [relation, key])
                counts[
                    "R6_info_type_release"
                    if relation == 6
                    else "R8_info_type_rating"
                ] += 1

        movie_kind_ids: set[int] = set()
        for row_no, row in enumerate(source.rows("kind_type"), start=1):
            require_columns(row, 2, "kind_type", row_no)
            if row[1] == "movie":
                key = as_int(row[0], "kind_type", row_no, 0)
                movie_kind_ids.add(key)
                write_int_row(out, [1, key])
                counts["R1_kind_type"] += 1

        us_company_ids: set[int] = set()
        for row_no, row in enumerate(source.rows("company_name"), start=1):
            require_columns(row, 3, "company_name", row_no)
            if row[2] == "[us]":
                key = as_int(row[0], "company_name", row_no, 0)
                us_company_ids.add(key)
                write_int_row(out, [3, key])
                counts["R3_company_name"] += 1

        if (
            not production_company_type_ids
            or not info_type_ids["release dates"]
            or not info_type_ids["rating"]
            or not movie_kind_ids
            or not us_company_ids
        ):
            raise ValueError(
                "JOB 13d predicate values were not found; use the original JOB IMDb snapshot"
            )

        sampled_title_ids: set[int] = set()
        selected_title_ids: set[int] = set()
        for row_no, row in enumerate(source.rows("title"), start=1):
            require_columns(row, 4, "title", row_no)
            movie_id = as_int(row[0], "title", row_no, 0)
            kind_id = as_int(row[3], "title", row_no, 3)
            if sample_percent < 100 and job13d_sample_bucket(movie_id) >= sample_percent:
                continue
            write_int_row(out, [0, movie_id, kind_id])
            counts["R0_title"] += 1
            sampled_title_ids.add(movie_id)
            if kind_id in movie_kind_ids:
                selected_title_ids.add(movie_id)

        selected_companies_by_movie: Counter[int] = Counter()
        for row_no, row in enumerate(source.rows("movie_companies"), start=1):
            require_columns(row, 4, "movie_companies", row_no)
            movie_id = as_int(row[1], "movie_companies", row_no, 1)
            company_id = as_int(row[2], "movie_companies", row_no, 2)
            company_type_id = as_int(row[3], "movie_companies", row_no, 3)
            if movie_id not in sampled_title_ids:
                continue
            write_int_row(out, [2, movie_id, company_id, company_type_id])
            counts["R2_movie_companies"] += 1
            if (
                company_id in us_company_ids
                and company_type_id in production_company_type_ids
            ):
                selected_companies_by_movie[movie_id] += 1

        release_rows_by_movie: Counter[int] = Counter()
        for row_no, row in enumerate(source.rows("movie_info"), start=1):
            require_columns(row, 3, "movie_info", row_no)
            movie_id = as_int(row[1], "movie_info", row_no, 1)
            info_type_id = as_int(row[2], "movie_info", row_no, 2)
            if movie_id not in sampled_title_ids:
                continue
            write_int_row(out, [5, movie_id, info_type_id])
            counts["R5_movie_info"] += 1
            if info_type_id in info_type_ids["release dates"]:
                release_rows_by_movie[movie_id] += 1

        rating_rows_by_movie: Counter[int] = Counter()
        for row_no, row in enumerate(source.rows("movie_info_idx"), start=1):
            require_columns(row, 3, "movie_info_idx", row_no)
            movie_id = as_int(row[1], "movie_info_idx", row_no, 1)
            info_type_id = as_int(row[2], "movie_info_idx", row_no, 2)
            if movie_id not in sampled_title_ids:
                continue
            write_int_row(out, [7, movie_id, info_type_id])
            counts["R7_movie_info_idx"] += 1
            if info_type_id in info_type_ids["rating"]:
                rating_rows_by_movie[movie_id] += 1

    expected = sum(
        company_rows
        * release_rows_by_movie.get(movie_id, 0)
        * rating_rows_by_movie.get(movie_id, 0)
        for movie_id, company_rows in selected_companies_by_movie.items()
        if movie_id in selected_title_ids
    )
    if expected <= 0 or expected > INT_MAX:
        raise ValueError(f"JOB 13d output {expected} is outside the benchmark int limit")

    # id, parent, join column in parent, join column in child, column count
    tree_rows = (
        (0, -1, -1, -1, 2),  # title(id, kind_id)
        (1, 0, 1, 0, 1),     # kind_type(id)
        (2, 0, 0, 0, 3),     # movie_companies(movie_id, company_id, type_id)
        (3, 2, 1, 0, 1),     # company_name(id), filtered to [us]
        (4, 2, 2, 0, 1),     # company_type(id), filtered to production
        (5, 0, 0, 0, 2),     # movie_info(movie_id, info_type_id)
        (6, 5, 1, 0, 1),     # info_type(id), filtered to release dates
        (7, 0, 0, 0, 2),     # movie_info_idx(movie_id, info_type_id)
        (8, 7, 1, 0, 1),     # info_type(id), filtered to rating
    )
    with (out_dir / "tree.txt").open("w", encoding="utf-8") as out:
        for row in tree_rows:
            write_int_row(out, row)
    (out_dir / "expected.txt").write_text(f"{expected}\n", encoding="utf-8")

    relation_names = (
        "R0_title",
        "R1_kind_type",
        "R2_movie_companies",
        "R3_company_name",
        "R4_company_type",
        "R5_movie_info",
        "R6_info_type_release",
        "R7_movie_info_idx",
        "R8_info_type_rating",
    )
    total_input = sum(counts[name] for name in relation_names)
    with (out_dir / "stats.txt").open("w", encoding="utf-8") as out:
        out.write("workload=JOB-13d join core before MIN aggregation\n")
        out.write("query_url=https://github.com/gregrahn/join-order-benchmark/blob/master/13d.sql\n")
        out.write(f"source={source_path.name}\n")
        out.write(f"sample_percent={sample_percent}\n")
        out.write("sample_key=title.id/movie_id\n")
        out.write("sample_rule=multiplicative-hash-bucket-mod-100\n")
        out.write("relations=9\n")
        out.write("depth=2\n")
        out.write("branching=4\n")
        out.write(f"input_rows={total_input}\n")
        out.write(f"output_rows={expected}\n")
        out.write(f"output_to_input_ratio={expected / total_input:.8f}\n")
        for name in relation_names:
            out.write(f"{name}={counts[name]}\n")
    print(
        f"[prepared] JOB 13d {sample_percent}%: {out_dir} "
        f"input={total_input} output={expected}",
        flush=True,
    )


def read_key_value_stats(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def export_job13d_samples_from_full(
    full_dir: Path,
    samples: list[tuple[int, Path]],
    force: bool,
) -> None:
    """Create nested JOB 13d samples in one pass over the full projection."""
    relation_names = (
        "R0_title",
        "R1_kind_type",
        "R2_movie_companies",
        "R3_company_name",
        "R4_company_type",
        "R5_movie_info",
        "R6_info_type_release",
        "R7_movie_info_idx",
        "R8_info_type_rating",
    )
    states: dict[int, dict[str, object]] = {}
    for percent, case_dir in samples:
        if percent < 1 or percent >= 100:
            raise ValueError("projected JOB samples must be in [1, 99]")
        prepare_output_dir(case_dir, force)
        states[percent] = {
            "dir": case_dir,
            "out": (case_dir / "tables.tbl").open("w", encoding="utf-8"),
            "counts": Counter(),
            "title_ids": set(),
            "selected_title_ids": set(),
            "companies": Counter(),
            "releases": Counter(),
            "ratings": Counter(),
        }

    movie_kind_ids: set[int] = set()
    us_company_ids: set[int] = set()
    production_company_type_ids: set[int] = set()
    release_info_type_ids: set[int] = set()
    rating_info_type_ids: set[int] = set()
    try:
        with (full_dir / "tables.tbl").open(encoding="utf-8") as source:
            for row_no, line in enumerate(source, start=1):
                fields = [int(value) for value in line.split()]
                if not fields:
                    continue
                relation = fields[0]
                row = fields[1:]
                if relation < 0 or relation >= len(relation_names):
                    raise ValueError(f"full JOB row {row_no} has invalid relation {relation}")

                if relation == 1:
                    movie_kind_ids.add(row[0])
                elif relation == 3:
                    us_company_ids.add(row[0])
                elif relation == 4:
                    production_company_type_ids.add(row[0])
                elif relation == 6:
                    release_info_type_ids.add(row[0])
                elif relation == 8:
                    rating_info_type_ids.add(row[0])

                for percent, state in states.items():
                    keep = relation in (1, 3, 4, 6, 8)
                    if relation == 0:
                        keep = job13d_sample_bucket(row[0]) < percent
                        if keep:
                            title_ids = state["title_ids"]
                            selected_title_ids = state["selected_title_ids"]
                            assert isinstance(title_ids, set)
                            assert isinstance(selected_title_ids, set)
                            title_ids.add(row[0])
                            if row[1] in movie_kind_ids:
                                selected_title_ids.add(row[0])
                    elif relation in (2, 5, 7):
                        title_ids = state["title_ids"]
                        assert isinstance(title_ids, set)
                        keep = row[0] in title_ids
                    if not keep:
                        continue

                    output = state["out"]
                    counts = state["counts"]
                    assert isinstance(output, io.TextIOBase)
                    assert isinstance(counts, Counter)
                    write_int_row(output, fields)
                    counts[relation_names[relation]] += 1

                    if relation == 2 and row[1] in us_company_ids \
                            and row[2] in production_company_type_ids:
                        companies = state["companies"]
                        assert isinstance(companies, Counter)
                        companies[row[0]] += 1
                    elif relation == 5 and row[1] in release_info_type_ids:
                        releases = state["releases"]
                        assert isinstance(releases, Counter)
                        releases[row[0]] += 1
                    elif relation == 7 and row[1] in rating_info_type_ids:
                        ratings = state["ratings"]
                        assert isinstance(ratings, Counter)
                        ratings[row[0]] += 1
    finally:
        for state in states.values():
            output = state["out"]
            assert isinstance(output, io.TextIOBase)
            output.close()

    full_stats = read_key_value_stats(full_dir / "stats.txt")
    for percent, state in states.items():
        case_dir = state["dir"]
        counts = state["counts"]
        selected_title_ids = state["selected_title_ids"]
        companies = state["companies"]
        releases = state["releases"]
        ratings = state["ratings"]
        assert isinstance(case_dir, Path)
        assert isinstance(counts, Counter)
        assert isinstance(selected_title_ids, set)
        assert isinstance(companies, Counter)
        assert isinstance(releases, Counter)
        assert isinstance(ratings, Counter)
        expected = sum(
            company_rows * releases.get(movie_id, 0) * ratings.get(movie_id, 0)
            for movie_id, company_rows in companies.items()
            if movie_id in selected_title_ids
        )
        if expected <= 0 or expected > INT_MAX:
            raise ValueError(f"JOB 13d {percent}% output {expected} is outside int32")
        shutil.copyfile(full_dir / "tree.txt", case_dir / "tree.txt")
        (case_dir / "expected.txt").write_text(f"{expected}\n", encoding="utf-8")
        total_input = sum(counts[name] for name in relation_names)
        with (case_dir / "stats.txt").open("w", encoding="utf-8") as out:
            out.write("workload=JOB-13d join core before MIN aggregation\n")
            out.write(f"query_url={full_stats['query_url']}\n")
            out.write(f"source={full_stats['source']}\n")
            out.write(f"sample_percent={percent}\n")
            out.write("sample_key=title.id/movie_id\n")
            out.write("sample_rule=multiplicative-hash-bucket-mod-100\n")
            out.write("relations=9\n")
            out.write("depth=2\n")
            out.write("branching=4\n")
            out.write(f"input_rows={total_input}\n")
            out.write(f"output_rows={expected}\n")
            out.write(f"output_to_input_ratio={expected / total_input:.8f}\n")
            for name in relation_names:
                out.write(f"{name}={counts[name]}\n")
        print(
            f"[prepared] JOB 13d {percent}%: {case_dir} "
            f"input={total_input} output={expected}",
            flush=True,
        )


def export_job13d_scales(
    source_path: Path,
    out_root: Path,
    full_dir: Path,
    force: bool,
) -> None:
    """Build a fixed-query, nested real-data scale series for JOB 13d."""
    out_root.mkdir(parents=True, exist_ok=True)
    required_full_files = ("tables.tbl", "tree.txt", "expected.txt", "stats.txt")
    if not all((full_dir / name).is_file() for name in required_full_files):
        print(f"[prepare] missing full JOB 13d data; generating {full_dir}", flush=True)
        export_job13d(source_path, full_dir, False, 100)
    else:
        print(f"[reuse] full JOB 13d data: {full_dir}", flush=True)
    case_dirs = [
        (percent, out_root / f"pct{percent}")
        for percent in (20, 40, 60, 80)
    ]
    export_job13d_samples_from_full(full_dir, case_dirs, force)
    case_dirs.append((100, full_dir))

    tree_reference = (case_dirs[0][1] / "tree.txt").read_text(encoding="utf-8")
    rows: list[dict[str, str | int | float]] = []
    previous_titles: set[int] | None = None
    for percent, case_dir in case_dirs:
        if (case_dir / "tree.txt").read_text(encoding="utf-8") != tree_reference:
            raise ValueError(f"JOB 13d tree differs at {percent}%")
        stats = read_key_value_stats(case_dir / "stats.txt")
        expected = int((case_dir / "expected.txt").read_text(encoding="utf-8").strip())
        if expected != int(stats["output_rows"]) or expected <= 0:
            raise ValueError(f"JOB 13d expected-row check failed at {percent}%")

        title_ids: set[int] = set()
        seen_root = False
        with (case_dir / "tables.tbl").open(encoding="utf-8") as source:
            for line in source:
                fields = line.split()
                if fields and fields[0] == "0":
                    seen_root = True
                    title_ids.add(int(fields[1]))
                elif seen_root:
                    break
        if previous_titles is not None and not previous_titles.issubset(title_ids):
            raise ValueError(f"JOB 13d samples are not nested at {percent}%")
        previous_titles = title_ids

        rows.append({
            "case": f"job13d_pct{percent}",
            "sample_percent": percent,
            "path": Path(os.path.relpath(case_dir, out_root)).as_posix(),
            "relations": int(stats["relations"]),
            "depth": int(stats["depth"]),
            "branching": int(stats["branching"]),
            "input_rows": int(stats["input_rows"]),
            "output_rows": expected,
            "output_to_input_ratio": float(stats["output_to_input_ratio"]),
        })

    suite_path = out_root / "suite.csv"
    with suite_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[verified] nested JOB 13d scale suite: {suite_path}", flush=True)


def load_snap_edges(path: Path) -> list[tuple[int, int]]:
    opener = gzip.open if path.suffix == ".gz" else open
    edges: list[tuple[int, int]] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for row_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"SNAP edge row {row_no} has fewer than two columns")
            source, target = int(fields[0]), int(fields[1])
            if source < -INT_MAX - 1 or source > INT_MAX or target < -INT_MAX - 1 or target > INT_MAX:
                raise ValueError(f"SNAP edge row {row_no} contains an id outside int32")
            edges.append((source, target))
    if not edges:
        raise ValueError(f"no edges found in {path}")
    return edges


def tree_metadata(
    parent: list[int],
) -> tuple[list[list[int]], list[int], list[int], list[int], int, int]:
    if not parent or parent[0] != -1:
        raise ValueError("relation 0 must be the root")
    children: list[list[int]] = [[] for _ in parent]
    depths = [0] * len(parent)
    for relation in range(1, len(parent)):
        p = parent[relation]
        if p < 0 or p >= relation:
            raise ValueError("relations must be listed in parent-before-child order")
        children[p].append(relation)
        depths[relation] = depths[p] + 1
    join_parent = [-1] + [1] * (len(parent) - 1)
    join_child = [-1] + [0] * (len(parent) - 1)
    widths = [2] * len(parent)
    return (
        children,
        join_parent,
        join_child,
        widths,
        max(depths),
        max(len(nodes) for nodes in children),
    )


def select_hub_edge_subset(
    edges: list[tuple[int, int]],
    hubs: int,
    root_cap: int,
    middle_cap: int,
    leaf_cap: int,
) -> list[tuple[int, int]]:
    """Select a deterministic high-degree stress subset of real email edges."""
    unique_edges = list(dict.fromkeys(edges))
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    for source, target in unique_edges:
        outgoing[source].append(target)
        incoming[target].append(source)

    candidates: list[tuple[int, int, list[int]]] = []
    for node, targets in outgoing.items():
        if node not in incoming:
            continue
        child_nodes = list(
            dict.fromkeys(target for target in targets if outgoing.get(target))
        )
        child_nodes.sort(key=lambda target: (-len(outgoing[target]), target))
        child_nodes = child_nodes[:middle_cap]
        if not child_nodes:
            continue
        bounded_score = sum(min(leaf_cap, len(outgoing[target])) ** 2 for target in child_nodes)
        candidates.append((bounded_score, node, child_nodes))
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    if len(candidates) < hubs:
        raise ValueError(f"SNAP input has only {len(candidates)} usable hub nodes")

    selected: set[tuple[int, int]] = set()
    for _score, node, child_nodes in candidates[:hubs]:
        selected.update((source, node) for source in incoming[node][:root_cap])
        for child_node in child_nodes:
            selected.add((node, child_node))
            selected.update(
                (child_node, target)
                for target in outgoing[child_node][:leaf_cap]
            )
    if not selected:
        raise ValueError("hub selection produced no edges")
    return sorted(selected)


def edge_alias_tables(
    edge_rows: list[tuple[int, int]], relation_count: int
) -> list[list[list[int]]]:
    table = [[source, target] for source, target in edge_rows]
    return [[list(row) for row in table] for _ in range(relation_count)]


def mixed_outdegree_alias_tables(
    core_edges: list[tuple[int, int]], degrees: list[int]
) -> list[list[list[int]]]:
    """Build real-edge aliases with a fixed per-source degree for each table."""
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in core_edges:
        outgoing[source].append(target)
    for targets in outgoing.values():
        targets.sort()
    maximum = max(degrees)
    if any(len(targets) < maximum for targets in outgoing.values()):
        raise ValueError("core does not support the requested alias degrees")
    return [
        [
            [source, target]
            for source in sorted(outgoing)
            for target in outgoing[source][:degree]
        ]
        for degree in degrees
    ]


def fixed_outdegree_core_edges(
    all_edges: list[tuple[int, int]], degree: int
) -> list[tuple[int, int]]:
    """Select a real directed core with exactly ``degree`` edges per source.

    The core is obtained by repeatedly removing nodes with fewer than
    ``degree`` outgoing neighbors inside the remaining node set.  Keeping
    exactly ``degree`` real edges for every surviving source makes every
    parent tuple have the same number of matches in every relation alias.
    Consequently, equal-size self-joins have the same exact output for every
    acyclic tree with the same number of relations.
    """
    if degree <= 0:
        raise ValueError("fixed outdegree must be positive")
    outgoing: dict[int, set[int]] = defaultdict(set)
    incoming: dict[int, set[int]] = defaultdict(set)
    nodes: set[int] = set()
    for source, target in all_edges:
        outgoing[source].add(target)
        incoming[target].add(source)
        nodes.add(source)
        nodes.add(target)

    active = set(nodes)
    active_degree = {
        node: sum(target in active for target in outgoing[node])
        for node in nodes
    }
    pending = deque(
        node for node in sorted(nodes) if active_degree[node] < degree
    )
    while pending:
        node = pending.popleft()
        if node not in active:
            continue
        active.remove(node)
        for predecessor in incoming[node]:
            if predecessor not in active:
                continue
            active_degree[predecessor] -= 1
            if active_degree[predecessor] == degree - 1:
                pending.append(predecessor)

    if not active:
        raise ValueError(f"SNAP input has no directed {degree}-out core")
    selected: list[tuple[int, int]] = []
    for source in sorted(active):
        targets = sorted(target for target in outgoing[source] if target in active)
        if len(targets) < degree:
            raise AssertionError("invalid fixed-outdegree core")
        selected.extend((source, target) for target in targets[:degree])
    return selected


def fixed_outdegree_scc_edges(
    all_edges: list[tuple[int, int]], degree: int, seed: int = 0
) -> list[tuple[int, int]]:
    """Return a deterministic strongly connected real-edge outdegree core.

    We first compute the maximal directed ``degree``-out core.  A seeded,
    deterministic sample keeps exactly ``degree`` outgoing real edges per
    node.  We then select the largest sink strongly connected component.
    Since it is a sink component, all selected outgoing edges stay inside it;
    every retained node therefore still has exactly ``degree`` matches.
    """
    if degree <= 0:
        raise ValueError("fixed outdegree must be positive")
    outgoing: dict[int, set[int]] = defaultdict(set)
    incoming: dict[int, set[int]] = defaultdict(set)
    nodes: set[int] = set()
    for source, target in all_edges:
        outgoing[source].add(target)
        incoming[target].add(source)
        nodes.add(source)
        nodes.add(target)

    active = set(nodes)
    active_degree = {
        node: sum(target in active for target in outgoing[node])
        for node in nodes
    }
    pending = deque(
        node for node in sorted(nodes) if active_degree[node] < degree
    )
    while pending:
        node = pending.popleft()
        if node not in active:
            continue
        active.remove(node)
        for predecessor in incoming[node]:
            if predecessor not in active:
                continue
            active_degree[predecessor] -= 1
            if active_degree[predecessor] == degree - 1:
                pending.append(predecessor)
    if not active:
        raise ValueError(f"SNAP input has no directed {degree}-out core")

    rng = random.Random(seed)
    chosen: dict[int, list[int]] = {}
    reverse: dict[int, list[int]] = defaultdict(list)
    for source in sorted(active):
        targets = sorted(target for target in outgoing[source] if target in active)
        selected = targets if len(targets) == degree else rng.sample(targets, degree)
        chosen[source] = selected
        for target in selected:
            reverse[target].append(source)

    # Iterative Kosaraju traversal avoids Python recursion limits.
    visited: set[int] = set()
    finish_order: list[int] = []
    for start in sorted(active):
        if start in visited:
            continue
        visited.add(start)
        stack: list[tuple[int, int]] = [(start, 0)]
        while stack:
            node, index = stack[-1]
            if index < degree:
                target = chosen[node][index]
                stack[-1] = (node, index + 1)
                if target not in visited:
                    visited.add(target)
                    stack.append((target, 0))
            else:
                finish_order.append(node)
                stack.pop()

    visited.clear()
    components: list[list[int]] = []
    component_id: dict[int, int] = {}
    for start in reversed(finish_order):
        if start in visited:
            continue
        component: list[int] = []
        visited.add(start)
        stack = [start]
        while stack:
            node = stack.pop()
            component.append(node)
            for predecessor in reverse[node]:
                if predecessor not in visited:
                    visited.add(predecessor)
                    stack.append(predecessor)
        identifier = len(components)
        for node in component:
            component_id[node] = identifier
        components.append(component)

    has_external_edge = [False] * len(components)
    for source, targets in chosen.items():
        for target in targets:
            if component_id[source] != component_id[target]:
                has_external_edge[component_id[source]] = True
    sink_ids = [
        identifier
        for identifier, has_external in enumerate(has_external_edge)
        if not has_external
    ]
    selected_id = max(sink_ids, key=lambda identifier: len(components[identifier]))
    selected_nodes = set(components[selected_id])
    return sorted(
        (source, target)
        for source in selected_nodes
        for target in chosen[source]
    )


def equal_output_relation_tables(
    all_edges: list[tuple[int, int]],
    base_edges: list[tuple[int, int]],
    relation_count: int,
) -> list[list[list[int]]]:
    """Build equal-cardinality real-edge tables for the relation-count series.

    Relations 0--3 use the same hub-edge table.  Every added leaf contains
    exactly one real outgoing edge for each reachable parent key.  Remaining
    rows are real edges whose source is never probed by those parents.  Thus
    adding leaves changes the number of relations and input rows, but neither
    the output cardinality nor the row count of an individual relation.
    """
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in all_edges:
        outgoing[source].append(target)

    base = [edge for edge in base_edges if edge[1] in outgoing]
    if not base:
        raise ValueError("relation-count base has no target with a real outgoing edge")
    target_keys = sorted({target for _source, target in base})
    functional = [(key, outgoing[key][0]) for key in target_keys]
    target_key_set = set(target_keys)
    padding = [edge for edge in all_edges if edge[0] not in target_key_set]
    needed = len(base) - len(functional)
    if needed < 0 or len(padding) < needed:
        raise ValueError("not enough non-matching real edges to equalize relation sizes")
    leaf = functional + padding[:needed]

    tables: list[list[list[int]]] = []
    for relation in range(relation_count):
        selected = base if relation <= 3 else leaf
        tables.append([[source, target] for source, target in selected])
    return tables


def exact_join_rows(
    tables: list[list[list[int]]],
    parent: list[int],
    children: list[list[int]],
    join_parent: list[int],
    join_child: list[int],
) -> int:
    contribution: list[list[int]] = [[] for _ in tables]
    for relation in range(len(tables) - 1, -1, -1):
        if not children[relation]:
            contribution[relation] = [1] * len(tables[relation])
            continue
        child_totals: dict[int, dict[int, int]] = {}
        for child in children[relation]:
            totals: dict[int, int] = defaultdict(int)
            column = join_child[child]
            for row, count in zip(tables[child], contribution[child]):
                totals[row[column]] += count
            child_totals[child] = totals
        relation_counts: list[int] = []
        for row in tables[relation]:
            count = 1
            for child in children[relation]:
                count *= child_totals[child].get(row[join_parent[child]], 0)
            relation_counts.append(count)
        contribution[relation] = relation_counts
    return sum(contribution[0])


def frequency_stats(keys: list[int]) -> tuple[float, int, int]:
    frequencies = sorted(Counter(keys).values())
    count = len(frequencies)
    total = sum(frequencies)
    weighted = sum((index + 1) * value for index, value in enumerate(frequencies))
    gini = (2.0 * weighted) / (count * total) - (count + 1.0) / count
    return gini, max(frequencies), count


def write_snap_case(
    case_dir: Path,
    force: bool,
    case_name: str,
    series: str,
    data_scale: str,
    selection: str,
    tables: list[list[list[int]]],
    parent: list[int],
    join_parent: list[int],
    join_child: list[int],
    widths: list[int],
    expected: int,
    source_name: str,
    control: str,
) -> dict[str, str | int | float]:
    if expected < 0 or expected > INT_MAX:
        raise ValueError(f"{case_name} output {expected} is outside the benchmark int limit")
    prepare_output_dir(case_dir, force)
    with (case_dir / "tree.txt").open("w", encoding="utf-8") as out:
        for relation in range(len(parent)):
            out.write(
                f"{relation} {parent[relation]} {join_parent[relation]} "
                f"{join_child[relation]} {widths[relation]}\n"
            )
    with (case_dir / "tables.tbl").open("w", encoding="utf-8") as out:
        for relation, table in enumerate(tables):
            for row in table:
                write_int_row(out, [relation, *row])
    (case_dir / "expected.txt").write_text(f"{expected}\n", encoding="utf-8")

    relation_count = len(parent)
    relation_row_counts = [len(table) for table in tables]
    input_rows = sum(len(table) for table in tables)
    source_keys = [row[0] for row in tables[0]]
    target_keys = [row[1] for row in tables[0]]
    source_gini, max_source_frequency, distinct_sources = frequency_stats(source_keys)
    target_gini, max_target_frequency, distinct_targets = frequency_stats(target_keys)
    children, _jp, _jc, _widths, depth, branching = tree_metadata(parent)
    del children, _jp, _jc, _widths
    stats = {
        "case": case_name,
        "series": series,
        "data_scale": data_scale,
        "branching": branching,
        "depth": depth,
        "relations": relation_count,
        "edge_rows_per_relation": len(tables[0]),
        "relation_row_counts": ";".join(map(str, relation_row_counts)),
        "min_relation_rows": min(relation_row_counts),
        "max_relation_rows": max(relation_row_counts),
        "input_rows": input_rows,
        "output_rows": expected,
        "selection": selection,
        "control": control,
        "source_key_gini": source_gini,
        "max_source_frequency": max_source_frequency,
        "distinct_sources": distinct_sources,
        "target_key_gini": target_gini,
        "max_target_frequency": max_target_frequency,
        "distinct_targets": distinct_targets,
        "source": source_name,
    }
    with (case_dir / "stats.txt").open("w", encoding="utf-8") as out:
        out.write("workload=real SNAP edge self-join with an acyclic query tree\n")
        out.write(f"source_url={SNAP_EMAIL_URL}\n")
        for key, value in stats.items():
            out.write(f"{key}={value}\n")
    return stats


def export_snap_acyclic_suite(
    source_path: Path,
    out_root: Path,
    force: bool,
    seed: int,
    max_output: int,
    hubs: int,
    root_cap: int,
    middle_cap: int,
    leaf_cap: int,
    relations_hubs: int,
    relations_root_cap: int,
    relations_middle_cap: int,
    relations_leaf_cap: int,
) -> None:
    if min(
        hubs, root_cap, middle_cap, leaf_cap,
        relations_hubs, relations_root_cap, relations_middle_cap,
        relations_leaf_cap, max_output,
    ) <= 0:
        raise ValueError("SNAP selection sizes and --max-output must be positive")
    edges = load_snap_edges(source_path)
    prepare_output_dir(out_root, force)
    unique_edges = list(dict.fromkeys(edges))
    hub_edges = select_hub_edge_subset(
        unique_edges, hubs, root_cap, middle_cap, leaf_cap
    )
    if len(hub_edges) > len(unique_edges):
        raise ValueError("hub edge selection exceeds the source size")
    relation_edges = select_hub_edge_subset(
        unique_edges,
        relations_hubs,
        relations_root_cap,
        relations_middle_cap,
        relations_leaf_cap,
    )
    depth_structure_edges = fixed_outdegree_scc_edges(unique_edges, 3, seed=0)
    branch_structure_edges = fixed_outdegree_scc_edges(unique_edges, 3, seed=1)
    relation_structure_edges = fixed_outdegree_scc_edges(unique_edges, 3, seed=2)

    # Depth and branching cases use nine relations and share the same tables
    # at each point. Relation-count cases use equal-sized tables and fixed output.
    depth_parents = {
        2: [-1, 0, 0, 0, 1, 1, 2, 2, 3],
        3: [-1, 0, 0, 0, 1, 1, 2, 2, 4],
        4: [-1, 0, 0, 0, 1, 1, 4, 4, 6],
    }
    branch_parents = {
        2: [-1, 0, 0, 1, 1, 2, 2, 3, 3],
        3: [-1, 0, 0, 0, 1, 1, 1, 4, 2],
        4: [-1, 0, 0, 0, 0, 1, 1, 1, 5],
    }
    cases: list[tuple[str, str, str, list[int], str, str, int | None]] = []
    three_degree_aliases = {
        "medium": {2: 0, 3: 2, 4: 4},
        "large": {2: 5, 3: 7, 4: 9},
    }
    for data_scale, aliases_by_factor in three_degree_aliases.items():
        control = (
            f"9 relations; {data_scale}-scale input/output grow with the tree factor"
        )
        for depth, parent in depth_parents.items():
            alias_count = aliases_by_factor[depth]
            cases.append((
                f"depth_d{depth}_m9_{data_scale}", "depth", data_scale,
                parent, f"mixed-2-3-out-scc-seed0-k{alias_count}", control,
                alias_count,
            ))
        for branching, parent in branch_parents.items():
            alias_count = aliases_by_factor[branching]
            cases.append((
                f"branch_b{branching}_m9_{data_scale}", "branching", data_scale,
                parent, f"mixed-2-3-out-scc-seed1-k{alias_count}", control,
                alias_count,
            ))
    cases.extend([
        ("relations_m5_d2_b3", "relations", "growing",
         [-1, 0, 0, 0, 1], "fixed-3-out-scc-seed2",
         "same real-edge selection; relations and output grow", None),
        ("relations_m7_d2_b3", "relations", "growing",
         [-1, 0, 0, 0, 1, 2, 3], "fixed-3-out-scc-seed2",
         "same real-edge selection; relations and output grow", None),
        ("relations_m9_d2_b3", "relations", "growing",
         [-1, 0, 0, 0, 1, 1, 2, 2, 3], "fixed-3-out-scc-seed2",
         "same real-edge selection; relations and output grow", None),
    ])

    summaries: list[dict[str, str | int | float]] = []
    for case_name, series, data_scale, parent, selection, control, alias_count in cases:
        children, join_parent, join_child, widths, _depth, _branching = tree_metadata(
            parent
        )
        if series == "relations":
            tables = edge_alias_tables(relation_structure_edges, len(parent))
        elif alias_count is not None:
            structure_edges = (
                depth_structure_edges if series == "depth"
                else branch_structure_edges
            )
            tables = mixed_outdegree_alias_tables(
                structure_edges,
                [3] * alias_count + [2] * (len(parent) - alias_count),
            )
        else:
            tables = edge_alias_tables(hub_edges, len(parent))
        expected = exact_join_rows(
            tables, parent, children, join_parent, join_child
        )
        if expected > max_output:
            raise ValueError(
                f"{case_name} output {expected} exceeds --max-output {max_output}"
            )
        summary = write_snap_case(
            out_root / case_name,
            force,
            case_name,
            series,
            data_scale,
            selection,
            tables,
            parent,
            join_parent,
            join_child,
            widths,
            expected,
            source_path.name,
            control,
        )
        summaries.append(summary)
        print(
            f"[prepared] {case_name}: relations={summary['relations']} "
            f"input={summary['input_rows']} output={expected}",
            flush=True,
        )

    fields = [
        "case",
        "series",
        "data_scale",
        "branching",
        "depth",
        "relations",
        "edge_rows_per_relation",
        "relation_row_counts",
        "min_relation_rows",
        "max_relation_rows",
        "input_rows",
        "output_rows",
        "selection",
        "control",
        "source_key_gini",
        "max_source_frequency",
        "distinct_sources",
        "target_key_gini",
        "max_target_frequency",
        "distinct_targets",
        "source",
    ]
    with (out_root / "suite.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[suite] {out_root / 'suite.csv'}", flush=True)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the official source when --input is missing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace generated files in a non-empty output directory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    job = subparsers.add_parser("job1a", help="prepare real IMDb JOB Query 1a")
    job.add_argument("--input", default="experiments/raw/imdb.tgz")
    job.add_argument("--out", default="experiments/data/job1a")
    add_common_arguments(job)

    job13d = subparsers.add_parser("job13d", help="prepare real IMDb JOB Query 13d")
    job13d.add_argument("--input", default="experiments/raw/imdb.tgz")
    job13d.add_argument("--out", default="experiments/data/job13d")
    job13d.add_argument(
        "--sample-percent",
        type=int,
        default=100,
        help="deterministic percentage of title/movie_id keys to retain (default: 100)",
    )
    add_common_arguments(job13d)

    job13d_scales = subparsers.add_parser(
        "job13d-scales",
        help="prepare nested 20%%, 40%%, 60%%, 80%%, and 100%% real IMDb JOB Query 13d inputs",
    )
    job13d_scales.add_argument("--input", default="experiments/raw/imdb.tgz")
    job13d_scales.add_argument("--out", default="experiments/data/job13d_scales")
    job13d_scales.add_argument("--full-out", default="experiments/data/job13d")
    add_common_arguments(job13d_scales)

    snap = subparsers.add_parser(
        "snap-acyclic",
        help="prepare real-edge acyclic query variants from SNAP email data",
    )
    snap.add_argument("--input", default="experiments/raw/email-EuAll.txt.gz")
    snap.add_argument("--out", default="experiments/data/snap_acyclic")
    snap.add_argument("--seed", type=int, default=7)
    snap.add_argument("--max-output", type=int, default=50_000_000)
    snap.add_argument("--hubs", type=int, default=15)
    snap.add_argument("--root-cap", type=int, default=6)
    snap.add_argument("--middle-cap", type=int, default=8)
    snap.add_argument("--leaf-cap", type=int, default=4)
    snap.add_argument("--relations-hubs", type=int, default=40)
    snap.add_argument("--relations-root-cap", type=int, default=10)
    snap.add_argument("--relations-middle-cap", type=int, default=10)
    snap.add_argument("--relations-leaf-cap", type=int, default=5)
    add_common_arguments(snap)

    tpch9 = subparsers.add_parser(
        "tpch9", help="generate and project the TPC-H Query 9 join tree"
    )
    tpch9.add_argument("--scale", default="0.1")
    tpch9.add_argument("--dbgen-dir", default="tpch/dbgen")
    tpch9.add_argument("--raw-dir", default="tpch/generated/sf_0p1")
    tpch9.add_argument("--out", default="tpch/sf_0p1")
    tpch9.add_argument(
        "--green-only",
        action="store_true",
        help="prepare the smaller SQL-predicate smoke test instead of the full join tree",
    )
    tpch9.add_argument(
        "--force",
        action="store_true",
        help="replace generated files in a non-empty output directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "job1a":
            source = require_input(Path(args.input).resolve(), JOB_URL, args.download)
            export_job1a(source, Path(args.out).resolve(), args.force)
        elif args.command == "job13d":
            source = require_input(Path(args.input).resolve(), JOB_URL, args.download)
            export_job13d(
                source,
                Path(args.out).resolve(),
                args.force,
                args.sample_percent,
            )
        elif args.command == "job13d-scales":
            source = require_input(Path(args.input).resolve(), JOB_URL, args.download)
            export_job13d_scales(
                source,
                Path(args.out).resolve(),
                Path(args.full_out).resolve(),
                args.force,
            )
        elif args.command == "snap-acyclic":
            source = require_input(
                Path(args.input).resolve(), SNAP_EMAIL_URL, args.download
            )
            export_snap_acyclic_suite(
                source,
                Path(args.out).resolve(),
                args.force,
                args.seed,
                args.max_output,
                args.hubs,
                args.root_cap,
                args.middle_cap,
                args.leaf_cap,
                args.relations_hubs,
                args.relations_root_cap,
                args.relations_middle_cap,
                args.relations_leaf_cap,
            )
        else:
            export_tpch9(
                Path(args.dbgen_dir).resolve(),
                Path(args.raw_dir).resolve(),
                Path(args.out).resolve(),
                args.scale,
                args.force,
                args.green_only,
            )
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
