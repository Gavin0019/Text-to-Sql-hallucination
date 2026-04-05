# (same imports unchanged)
import argparse
import json
import re
import heapq
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    sqlglot = None
    exp = None


# ------------------------
# existing functions (unchanged)
# ------------------------

def extract_schema_and_question(input_seq: str) -> Tuple[str, str]:
    schema_match = re.search(
        r"Database Schema:\s*(.*?)\s*This schema describes",
        input_seq,
        flags=re.DOTALL | re.IGNORECASE,
    )
    question_match = re.search(
        r"Question:\s*(.*?)\s*Instructions:",
        input_seq,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return (
        schema_match.group(1).strip() if schema_match else "",
        question_match.group(1).strip() if question_match else "",
    )


def parse_schema_from_ddl(schema_ddl: str) -> Dict[str, Set[str]]:
    schema = {}
    table_blocks = re.findall(
        r"CREATE TABLE\s+[`\"]?(\w+)[`\"]?\s*\((.*?)\);",
        schema_ddl,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for table_name, body in table_blocks:
        if table_name.lower().startswith("sqlite_"):
            continue
        cols = set()
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            if line.upper().startswith(("PRIMARY KEY", "FOREIGN KEY", "CONSTRAINT")):
                continue
            m = re.match(r"[`\"]?(\w+)[`\"]?\s+", line)
            if m:
                cols.add(m.group(1).lower())
        schema[table_name.lower()] = cols
    return schema


def extract_table_names_from_ddl(schema_ddl: str) -> Set[str]:
    return {
        t.lower()
        for t in re.findall(
            r"CREATE TABLE\s+[`\"]?(\w+)[`\"]?\s*\(",
            schema_ddl,
            flags=re.IGNORECASE,
        )
        if not t.lower().startswith("sqlite_")
    }


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().strip("`"))


# ------------------------
# extraction (same)
# ------------------------

def extract_used_schema(sql: str, schema: Dict[str, Set[str]], stats=None):
    sql = normalize_sql(sql).lower()
    used_tables, used_columns = set(), set()

    if sqlglot and exp:
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
            alias_map = {}

            for t in tree.find_all(exp.Table):
                name = t.name.lower()
                if name in schema:
                    used_tables.add(name)
                    alias_map[t.alias_or_name.lower()] = name

            for col in tree.find_all(exp.Column):
                c = col.name.lower()
                t = (col.table or "").lower()
                t = alias_map.get(t, t)
                if t in schema and c in schema[t]:
                    used_tables.add(t)
                    used_columns.add(f"{t}.{c}")

            if stats:
                stats["sqlglot_used"] = stats.get("sqlglot_used", 0) + 1
            return {"tables": used_tables, "columns": used_columns}
        except:
            pass

    # fallback
    for t in re.findall(r"\b(from|join)\s+(\w+)", sql):
        if t[1] in schema:
            used_tables.add(t[1])

    return {"tables": used_tables, "columns": used_columns}


# ------------------------
# FIXED METRICS (important)
# ------------------------

def safe_div(a, b):
    return a / b if b else 0.0


def precision(pred, gold):
    if not pred and not gold:
        return 1.0
    return safe_div(len(pred & gold), len(pred))


def recall(pred, gold):
    if not pred and not gold:
        return 1.0
    return safe_div(len(pred & gold), len(gold))


def f1(pred, gold):
    if not pred and not gold:
        return 1.0
    p, r = precision(pred, gold), recall(pred, gold)
    return safe_div(2 * p * r, p + r)


def hallucination_rate(pred, gold):
    return safe_div(len(pred - gold), len(pred))


def missing_rate(pred, gold):
    return safe_div(len(gold - pred), len(gold))


# ------------------------
# NEW: subgraph edges
# ------------------------

def extract_table_edges(used_tables, table_graph):
    edges = set()
    nodes = {f"t:{t}" for t in used_tables}
    for n in nodes:
        for neigh in table_graph.get(n, []):
            if neigh in nodes:
                a = n.split(":")[1]
                b = neigh.split(":")[1]
                edges.add(tuple(sorted((a, b))))
    return edges


# ------------------------
# NEW: repair metric
# ------------------------

def min_added_nodes_between(graph, pred_nodes, start, end):
    pq = [(0, start)]
    best = {start: 0}

    while pq:
        cost, node = heapq.heappop(pq)
        if node == end:
            return cost
        for nxt in graph.get(node, []):
            add = 0 if nxt in pred_nodes else 1
            new = cost + add
            if new < best.get(nxt, 999):
                best[nxt] = new
                heapq.heappush(pq, (new, nxt))
    return None


def compute_repair(gold_tables, pred_tables, graph):
    gold_nodes = {f"t:{t}" for t in gold_tables}
    pred_nodes = {f"t:{t}" for t in pred_tables}

    costs = []
    for a, b in combinations(gold_nodes, 2):
        c = min_added_nodes_between(graph, pred_nodes, a, b)
        if c is not None:
            costs.append(c)

    if not costs:
        return 1.0

    avg = sum(costs) / len(costs)
    return 1 - avg / (avg + 2)


# ------------------------
# MAIN METRICS
# ------------------------

def compute_metrics(gold_used, pred_used, table_graph,
                    gold_table_paths, pred_table_paths,
                    gold_column_paths, pred_column_paths):

    gold_tables = gold_used["tables"]
    pred_tables = pred_used["tables"]

    gold_cols = gold_used["columns"]
    pred_cols = pred_used["columns"]

    gold_schema = gold_tables | gold_cols
    pred_schema = pred_tables | pred_cols

    # existing
    gold_kg = gold_table_paths | gold_column_paths
    pred_kg = pred_table_paths | pred_column_paths

    # NEW
    gold_edges = extract_table_edges(gold_tables, table_graph)
    pred_edges = extract_table_edges(pred_tables, table_graph)

    return {
        # existing metrics
        "table_f1": f1(pred_tables, gold_tables),
        "column_f1": f1(pred_cols, gold_cols),
        "schema_f1": f1(pred_schema, gold_schema),
        "kg_path_f1": f1(pred_kg, gold_kg),

        # NEW metrics
        "subgraph_edge_f1": f1(pred_edges, gold_edges),
        "subgraph_edge_precision": precision(pred_edges, gold_edges),
        "subgraph_edge_recall": recall(pred_edges, gold_edges),

        "subgraph_exact_match": float(
            pred_tables == gold_tables and pred_edges == gold_edges
        ),

        "subgraph_repair_score": compute_repair(
            gold_tables, pred_tables, table_graph
        )
    }


# ------------------------
# evaluate loop (minimal change)
# ------------------------

def evaluate_file(path, db_to_tables, table_graph_by_db, column_graph_by_db):
    data = json.load(open(path))

    results = []

    for row in data:
        schema_ddl, question = extract_schema_and_question(row["input_seq"])
        schema = parse_schema_from_ddl(schema_ddl)

        db_id = list(db_to_tables.keys())[0]  # simplified

        table_graph = table_graph_by_db.get(db_id, {})

        gold = extract_used_schema(row["output_seq"], schema)
        pred = extract_used_schema(row["pred_sqls"][0], schema)

        # keep original paths
        gold_table_paths = set()
        pred_table_paths = set()
        gold_column_paths = set()
        pred_column_paths = set()

        metrics = compute_metrics(
            gold, pred, table_graph,
            gold_table_paths, pred_table_paths,
            gold_column_paths, pred_column_paths
        )

        results.append(metrics)

    summary = {}
    for k in results[0]:
        summary[k] = sum(r[k] for r in results) / len(results)

    return summary


def parse_node(node_id: str) -> Tuple[str, str]:
    # returns (db_id, local_id)
    if node_id.startswith("db:"):
        return node_id.split(":", 1)[1], ""
    if node_id.startswith("table:"):
        payload = node_id.split(":", 1)[1]
        db_id, table = payload.split(".", 1)
        return db_id, f"t:{table.lower()}"
    if node_id.startswith("column:"):
        payload = node_id.split(":", 1)[1]
        db_id, rest = payload.split(".", 1)
        table, col = rest.split(".", 1)
        return db_id, f"c:{table.lower()}.{col.lower()}"
    return "", ""


def load_global_kg(kg_path: Path):
    with kg_path.open("r", encoding="utf-8") as f:
        kg = json.load(f)

    db_to_tables: Dict[str, Set[str]] = {}
    table_graph_by_db: Dict[str, Dict[str, Set[str]]] = {}
    column_graph_by_db: Dict[str, Dict[str, Set[str]]] = {}

    nodes_by_id = {n["id"]: n for n in kg["nodes"]}

    for node in kg["nodes"]:
        if node.get("type") == "table":
            db_id = node["db_id"]
            table = node["table"].lower()
            db_to_tables.setdefault(db_id, set()).add(table)
            table_graph_by_db.setdefault(db_id, {}).setdefault(f"t:{table}", set())
            column_graph_by_db.setdefault(db_id, {}).setdefault(f"t:{table}", set())
        elif node.get("type") == "column":
            db_id = node["db_id"]
            col = f"c:{node['table'].lower()}.{node['column'].lower()}"
            column_graph_by_db.setdefault(db_id, {}).setdefault(col, set())

    for edge in kg["edges"]:
        src_id, dst_id, rel = edge["source"], edge["target"], edge["relation"]
        src_db, src_local = parse_node(src_id)
        dst_db, dst_local = parse_node(dst_id)
        if not src_db or not dst_db or src_db != dst_db:
            continue
        db_id = src_db

        if rel == "RELATES_TO" and src_local.startswith("t:") and dst_local.startswith("t:"):
            tg = table_graph_by_db.setdefault(db_id, {})
            tg.setdefault(src_local, set()).add(dst_local)
            tg.setdefault(dst_local, set()).add(src_local)

        if rel in {"HAS_COLUMN", "PRIMARY_KEY", "FK_TO"}:
            cg = column_graph_by_db.setdefault(db_id, {})
            if src_local and dst_local:
                cg.setdefault(src_local, set()).add(dst_local)
                cg.setdefault(dst_local, set()).add(src_local)

        # Safety for missing RELATES_TO: infer table-table links from FK_TO columns.
        if rel == "FK_TO" and src_local.startswith("c:") and dst_local.startswith("c:"):
            src_table = "t:" + src_local.split(":", 1)[1].split(".", 1)[0]
            dst_table = "t:" + dst_local.split(":", 1)[1].split(".", 1)[0]
            tg = table_graph_by_db.setdefault(db_id, {})
            tg.setdefault(src_table, set()).add(dst_table)
            tg.setdefault(dst_table, set()).add(src_table)

    return db_to_tables, table_graph_by_db, column_graph_by_db

# ------------------------
# MAIN
# ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="*_dev_spider_syn.json")
    parser.add_argument("--kg-file", default="spider_dev_only_schema_kg.json")
    args = parser.parse_args()

    db_to_tables, table_graph_by_db, column_graph_by_db = load_global_kg(Path(args.kg_file))

    files = sorted(Path(".").glob(args.input_glob))

    for f in files:
        summary = evaluate_file(f, db_to_tables, table_graph_by_db, column_graph_by_db)
        print(f"\n{f}")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}")