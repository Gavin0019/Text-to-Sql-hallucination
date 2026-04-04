import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def discover_sqlite_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.sqlite", "*.db", "*.sqlite3"):
        files.extend(root.rglob(ext))
    return sorted(set(files))


def load_allowed_db_ids(dev_json_path: Path) -> Set[str]:
    with dev_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {row["db_id"] for row in data if "db_id" in row}


def table_node_id(db_id: str, table: str) -> str:
    return f"table:{db_id}.{table}"


def column_node_id(db_id: str, table: str, column: str) -> str:
    return f"column:{db_id}.{table}.{column}"


def build_kg(database_root: Path, allowed_db_ids: Set[str] | None = None) -> Dict:
    sqlite_files = discover_sqlite_files(database_root)
    if allowed_db_ids is not None:
        sqlite_files = [p for p in sqlite_files if p.stem in allowed_db_ids]

    nodes: List[Dict] = []
    edges: List[Dict] = []

    node_seen: Set[str] = set()
    edge_seen: Set[Tuple[str, str, str]] = set()

    def add_node(node: Dict) -> None:
        node_id = node["id"]
        if node_id in node_seen:
            return
        node_seen.add(node_id)
        nodes.append(node)

    def add_edge(source: str, relation: str, target: str, extra: Dict | None = None) -> None:
        key = (source, relation, target)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edge = {
            "source": source,
            "relation": relation,
            "target": target,
        }
        if extra:
            edge.update(extra)
        edges.append(edge)

    for db_path in sqlite_files:
        db_id = db_path.stem
        db_node = f"db:{db_id}"
        add_node(
            {
                "id": db_node,
                "type": "database",
                "db_id": db_id,
                "path": str(db_path),
            }
        )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            table_names = [r["name"] for r in table_rows]

            for table in table_names:
                table_id = table_node_id(db_id, table)
                add_node(
                    {
                        "id": table_id,
                        "type": "table",
                        "db_id": db_id,
                        "table": table,
                    }
                )
                add_edge(db_node, "HAS_TABLE", table_id)

                pragma_table = quote_ident(table)
                col_rows = conn.execute(f"PRAGMA table_info({pragma_table})").fetchall()
                for col in col_rows:
                    col_name = col["name"]
                    col_id = column_node_id(db_id, table, col_name)
                    add_node(
                        {
                            "id": col_id,
                            "type": "column",
                            "db_id": db_id,
                            "table": table,
                            "column": col_name,
                            "data_type": col["type"],
                            "not_null": bool(col["notnull"]),
                            "is_primary_key": bool(col["pk"]),
                        }
                    )
                    add_edge(table_id, "HAS_COLUMN", col_id)
                    if col["pk"]:
                        add_edge(table_id, "PRIMARY_KEY", col_id)

                fk_rows = conn.execute(f"PRAGMA foreign_key_list({pragma_table})").fetchall()
                for fk in fk_rows:
                    src_col = fk["from"]
                    tgt_table = fk["table"]
                    tgt_col = fk["to"]
                    if not tgt_col:
                        continue

                    src_col_id = column_node_id(db_id, table, src_col)
                    tgt_col_id = column_node_id(db_id, tgt_table, tgt_col)
                    src_table_id = table_node_id(db_id, table)
                    tgt_table_id = table_node_id(db_id, tgt_table)

                    add_edge(
                        src_col_id,
                        "FK_TO",
                        tgt_col_id,
                        extra={"fk_id": fk["id"], "on_update": fk["on_update"], "on_delete": fk["on_delete"]},
                    )
                    add_edge(src_table_id, "RELATES_TO", tgt_table_id)
        finally:
            conn.close()

    return {
        "meta": {
            "database_root": str(database_root),
            "num_sqlite_files": len(sqlite_files),
            "filtered_by_db_ids": sorted(allowed_db_ids) if allowed_db_ids is not None else None,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a merged schema KG from many SQLite files.")
    parser.add_argument("--database-root", default="database", help="Root folder containing SQLite files")
    parser.add_argument("--output", default="spider_dev_schema_kg.json", help="Output KG JSON path")
    parser.add_argument(
        "--dev-json",
        default=None,
        help="Optional dev.json path; when set, only db_ids appearing in this file are included",
    )
    args = parser.parse_args()

    database_root = Path(args.database_root)
    allowed_db_ids = None
    if args.dev_json:
        allowed_db_ids = load_allowed_db_ids(Path(args.dev_json))

    kg = build_kg(database_root, allowed_db_ids=allowed_db_ids)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

    print(f"Wrote KG to: {output_path}")
    for k, v in kg["meta"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
