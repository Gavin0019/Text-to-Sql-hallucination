import argparse
import heapq
import json
import math
import re
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
    schema_ddl = schema_match.group(1).strip() if schema_match else ""
    question = question_match.group(1).strip() if question_match else ""
    return schema_ddl, question


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
            upper = line.upper()
            if upper.startswith("PRIMARY KEY") or upper.startswith("FOREIGN KEY") or upper.startswith("CONSTRAINT"):
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
    sql = sql.strip()
    sql = sql.strip("`")
    sql = re.sub(r"\s+", " ", sql)
    return sql


def extract_used_schema(
    sql: str,
    schema: Dict[str, Set[str]],
    stats: Dict[str, int] | None = None,
) -> Dict[str, Set[str]]:
    sql_norm = normalize_sql(sql).lower()
    used_tables = set()
    used_columns = set()

    if sqlglot is not None and exp is not None:
        try:
            tree = sqlglot.parse_one(sql_norm, read="sqlite")
            alias_to_table = {}
            for table in tree.find_all(exp.Table):
                table_name = table.name.lower() if table.name else ""
                if not table_name:
                    continue
                if table_name in schema:
                    used_tables.add(table_name)
                    alias = (table.alias_or_name or "").lower()
                    if alias:
                        alias_to_table[alias] = table_name
                    alias_to_table[table_name] = table_name

            for col in tree.find_all(exp.Column):
                col_name = (col.name or "").lower()
                if not col_name:
                    continue
                qualifier = (col.table or "").lower()
                if qualifier:
                    table_name = alias_to_table.get(qualifier, qualifier)
                    if table_name in schema and col_name in schema[table_name]:
                        used_tables.add(table_name)
                        used_columns.add(f"{table_name}.{col_name}")
                else:
                    candidate_tables = [t for t in used_tables if col_name in schema[t]]
                    if len(candidate_tables) == 1:
                        used_columns.add(f"{candidate_tables[0]}.{col_name}")

            if stats is not None:
                stats["sqlglot_used"] = stats.get("sqlglot_used", 0) + 1
            return {"tables": used_tables, "columns": used_columns}
        except Exception:
            if stats is not None:
                stats["fallback_parse_error"] = stats.get("fallback_parse_error", 0) + 1
    else:
        if stats is not None:
            stats["fallback_sqlglot_unavailable"] = stats.get("fallback_sqlglot_unavailable", 0) + 1

    table_patterns = re.findall(r"\b(?:from|join|update|into)\s+([a-zA-Z_][\w]*)", sql_norm)
    for t in table_patterns:
        if t in schema:
            used_tables.add(t)

    for table in schema:
        if re.search(rf"\b{re.escape(table)}\b", sql_norm):
            used_tables.add(table)

    qualified_cols = re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", sql_norm)
    for t, c in qualified_cols:
        t = t.lower()
        c = c.lower()
        if t in schema and c in schema[t]:
            used_tables.add(t)
            used_columns.add(f"{t}.{c}")

    token_candidates = set(re.findall(r"\b[a-zA-Z_][\w]*\b", sql_norm))
    sql_keywords = {
        "select", "from", "where", "group", "by", "order", "limit", "join", "on",
        "and", "or", "not", "as", "avg", "min", "max", "count", "sum", "distinct",
        "having", "desc", "asc", "in", "exists", "like", "between", "union",
        "intersect", "except", "case", "when", "then", "else", "end"
    }
    for token in token_candidates:
        if token in sql_keywords:
            continue
        candidate_tables = [t for t in used_tables if token in schema[t]]
        if len(candidate_tables) == 1:
            used_columns.add(f"{candidate_tables[0]}.{token}")

    return {"tables": used_tables, "columns": used_columns}


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def precision(pred: Set[str], gold: Set[str]) -> float:
    return safe_div(len(pred & gold), len(pred))


def recall(pred: Set[str], gold: Set[str]) -> float:
    return safe_div(len(pred & gold), len(gold))


def f1(pred: Set[str], gold: Set[str]) -> float:
    p = precision(pred, gold)
    r = recall(pred, gold)
    return safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0


def hallucination_rate(pred: Set[str], gold: Set[str]) -> float:
    return safe_div(len(pred - gold), len(pred))


def missing_rate(pred: Set[str], gold: Set[str]) -> float:
    return safe_div(len(gold - pred), len(gold))


def shortest_path(graph: Dict[str, Set[str]], start: str, end: str) -> List[str]:
    if start == end:
        return [start]
    if start not in graph or end not in graph:
        return []
    q = deque([(start, [start])])
    visited = {start}
    while q:
        node, path = q.popleft()
        for nxt in sorted(graph.get(node, [])):
            if nxt in visited:
                continue
            next_path = path + [nxt]
            if nxt == end:
                return next_path
            visited.add(nxt)
            q.append((nxt, next_path))
    return []


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


def infer_db_id(schema_tables: Set[str], db_to_tables: Dict[str, Set[str]]) -> str | None:
    exact = [db for db, tables in db_to_tables.items() if tables == schema_tables]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return sorted(exact)[0]
    subset = [db for db, tables in db_to_tables.items() if schema_tables.issubset(tables)]
    if len(subset) == 1:
        return subset[0]
    if len(subset) > 1:
        return sorted(subset, key=lambda x: len(db_to_tables[x]))[0]
    return None


def extract_table_path_units(used_tables: Set[str], table_graph: Dict[str, Set[str]]) -> Set[str]:
    units: Set[str] = set()
    table_nodes = [f"t:{t}" for t in sorted(used_tables)]
    for t in table_nodes:
        units.add("table:" + t.split(":", 1)[1])
    for a, b in combinations(table_nodes, 2):
        path = shortest_path(table_graph, a, b)
        if path:
            pretty = [p.split(":", 1)[1] for p in path]
            units.add("table_path:" + ">".join(pretty))
    return units


def extract_column_path_units(used_columns: Set[str], column_graph: Dict[str, Set[str]]) -> Set[str]:
    units: Set[str] = set()
    col_nodes = [f"c:{c}" for c in sorted(used_columns)]
    for c in col_nodes:
        units.add("column:" + c.split(":", 1)[1])
    for a, b in combinations(col_nodes, 2):
        path = shortest_path(column_graph, a, b)
        if path:
            pretty = [p.split(":", 1)[1] if ":" in p else p for p in path]
            units.add("column_path:" + ">".join(pretty))
    return units


def extract_table_edges(used_tables: Set[str], table_graph: Dict[str, Set[str]]) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    nodes = {f"t:{t}" for t in used_tables}
    for n in nodes:
        for neigh in table_graph.get(n, []):
            if neigh in nodes:
                a = n.split(":", 1)[1]
                b = neigh.split(":", 1)[1]
                edges.add(tuple(sorted((a, b))))
    return edges


def min_added_nodes_between(
    graph: Dict[str, Set[str]], pred_nodes: Set[str], start: str, end: str
) -> int | None:
    pq: List[Tuple[int, str]] = [(0, start)]
    best: Dict[str, int] = {start: 0}

    while pq:
        cost, node = heapq.heappop(pq)
        if node == end:
            return cost
        for nxt in graph.get(node, []):
            add = 0 if nxt in pred_nodes else 1
            new_cost = cost + add
            if new_cost < best.get(nxt, 999):
                best[nxt] = new_cost
                heapq.heappush(pq, (new_cost, nxt))
    return None


def average_add_repair_cost(
    gold_tables: Set[str], pred_tables: Set[str], graph: Dict[str, Set[str]]
) -> float:
    gold_nodes = {f"t:{t}" for t in gold_tables}
    pred_nodes = {f"t:{t}" for t in pred_tables}

    costs: List[int] = []
    for a, b in combinations(gold_nodes, 2):
        c = min_added_nodes_between(graph, pred_nodes, a, b)
        if c is not None:
            costs.append(c)

    if not costs:
        return 0.0

    return sum(costs) / len(costs)


def map_repair_cost_to_score(cost: float) -> float:
    return math.exp(-cost)


def compute_hallucination_remove_cost(
    gold_tables: Set[str],
    pred_tables: Set[str],
    gold_edges: Set[Tuple[str, str]],
    pred_edges: Set[Tuple[str, str]],
) -> Tuple[int, int]:
    extra_tables = pred_tables - gold_tables
    n_tab = len(extra_tables)
    extra_edges = pred_edges - gold_edges
    n_edge = len(extra_edges)
    return n_tab, n_edge


def _edge_precision(pred: Set[Tuple[str, str]], gold: Set[Tuple[str, str]]) -> float:
    if not pred and not gold:
        return 1.0
    return safe_div(len(pred & gold), len(pred))


def _edge_recall(pred: Set[Tuple[str, str]], gold: Set[Tuple[str, str]]) -> float:
    if not pred and not gold:
        return 1.0
    return safe_div(len(pred & gold), len(gold))


def _edge_f1(pred: Set[Tuple[str, str]], gold: Set[Tuple[str, str]]) -> float:
    if not pred and not gold:
        return 1.0
    p, r = _edge_precision(pred, gold), _edge_recall(pred, gold)
    return safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0


def compute_subgraph_metrics(
    gold_used: Dict[str, Set[str]],
    pred_used: Dict[str, Set[str]],
    table_graph: Dict[str, Set[str]],
) -> Dict[str, float]:
    gold_tables = gold_used["tables"]
    pred_tables = pred_used["tables"]
    gold_edges = extract_table_edges(gold_tables, table_graph)
    pred_edges = extract_table_edges(pred_tables, table_graph)
    add_raw = average_add_repair_cost(gold_tables, pred_tables, table_graph)
    n_extra_tab, n_extra_edge = compute_hallucination_remove_cost(
        gold_tables, pred_tables, gold_edges, pred_edges
    )
    remove_raw = float(n_extra_tab + n_extra_edge)
    total_cost = add_raw + remove_raw
    return {
        "subgraph_edge_f1": _edge_f1(pred_edges, gold_edges),
        "subgraph_edge_precision": _edge_precision(pred_edges, gold_edges),
        "subgraph_edge_recall": _edge_recall(pred_edges, gold_edges),
        "subgraph_exact_match": float(
            pred_tables == gold_tables and pred_edges == gold_edges
        ),
        "subgraph_repair_add_raw": add_raw,
        "subgraph_repair_remove_raw": remove_raw,
        "subgraph_repair_add_score": map_repair_cost_to_score(add_raw),
        "subgraph_repair_remove_score": map_repair_cost_to_score(remove_raw),
        "subgraph_repair_extra_tables": float(n_extra_tab),
        "subgraph_repair_extra_edges": float(n_extra_edge),
        "subgraph_repair_score": map_repair_cost_to_score(total_cost),
    }


def compute_metrics(
    gold_used: Dict[str, Set[str]],
    pred_used: Dict[str, Set[str]],
    gold_table_paths: Set[str],
    pred_table_paths: Set[str],
    gold_column_paths: Set[str],
    pred_column_paths: Set[str],
) -> Dict[str, float]:
    gold_tables = gold_used["tables"]
    pred_tables = pred_used["tables"]
    gold_cols = gold_used["columns"]
    pred_cols = pred_used["columns"]
    gold_schema = gold_tables | gold_cols
    pred_schema = pred_tables | pred_cols

    gold_kg = gold_table_paths | gold_column_paths
    pred_kg = pred_table_paths | pred_column_paths

    return {
        "table_precision": precision(pred_tables, gold_tables),
        "table_recall": recall(pred_tables, gold_tables),
        "table_f1": f1(pred_tables, gold_tables),
        "table_hallucination_rate": hallucination_rate(pred_tables, gold_tables),
        "table_missing_rate": missing_rate(pred_tables, gold_tables),
        "column_precision": precision(pred_cols, gold_cols),
        "column_recall": recall(pred_cols, gold_cols),
        "column_f1": f1(pred_cols, gold_cols),
        "column_hallucination_rate": hallucination_rate(pred_cols, gold_cols),
        "column_missing_rate": missing_rate(pred_cols, gold_cols),
        "schema_precision": precision(pred_schema, gold_schema),
        "schema_recall": recall(pred_schema, gold_schema),
        "schema_f1": f1(pred_schema, gold_schema),
        "schema_hallucination_rate": hallucination_rate(pred_schema, gold_schema),
        "schema_missing_rate": missing_rate(pred_schema, gold_schema),
        "kg_table_path_precision": precision(pred_table_paths, gold_table_paths),
        "kg_table_path_recall": recall(pred_table_paths, gold_table_paths),
        "kg_table_path_f1": f1(pred_table_paths, gold_table_paths),
        "kg_table_path_hallucination_rate": hallucination_rate(pred_table_paths, gold_table_paths),
        "kg_table_path_missing_rate": missing_rate(pred_table_paths, gold_table_paths),
        "kg_column_path_precision": precision(pred_column_paths, gold_column_paths),
        "kg_column_path_recall": recall(pred_column_paths, gold_column_paths),
        "kg_column_path_f1": f1(pred_column_paths, gold_column_paths),
        "kg_column_path_hallucination_rate": hallucination_rate(pred_column_paths, gold_column_paths),
        "kg_column_path_missing_rate": missing_rate(pred_column_paths, gold_column_paths),
        "kg_path_precision": precision(pred_kg, gold_kg),
        "kg_path_recall": recall(pred_kg, gold_kg),
        "kg_path_f1": f1(pred_kg, gold_kg),
        "kg_path_hallucination_rate": hallucination_rate(pred_kg, gold_kg),
        "kg_path_missing_rate": missing_rate(pred_kg, gold_kg),
    }


def evaluate_file(
    path: str | Path,
    db_to_tables: Dict[str, Set[str]],
    table_graph_by_db: Dict[str, Dict[str, Set[str]]],
    column_graph_by_db: Dict[str, Dict[str, Set[str]]],
):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    parser_stats = {
        "sqlglot_used": 0,
        "fallback_parse_error": 0,
        "fallback_sqlglot_unavailable": 0,
        "db_id_infer_failed": 0,
    }

    for row in data:
        schema_ddl, question = extract_schema_and_question(row["input_seq"])
        schema = parse_schema_from_ddl(schema_ddl)
        schema_tables_for_match = extract_table_names_from_ddl(schema_ddl)
        db_id = infer_db_id(schema_tables_for_match, db_to_tables)
        if db_id is None:
            parser_stats["db_id_infer_failed"] += 1
            continue

        table_graph = table_graph_by_db.get(db_id, {})
        column_graph = column_graph_by_db.get(db_id, {})

        gold_sql = row["output_seq"].strip() if row.get("output_seq") else ""
        pred_sql = row["pred_sqls"][0].strip() if row.get("pred_sqls") else ""

        gold_used = extract_used_schema(gold_sql, schema, stats=parser_stats)
        pred_used = extract_used_schema(pred_sql, schema, stats=parser_stats)

        gold_table_paths = extract_table_path_units(gold_used["tables"], table_graph)
        pred_table_paths = extract_table_path_units(pred_used["tables"], table_graph)
        gold_column_paths = extract_column_path_units(gold_used["columns"], column_graph)
        pred_column_paths = extract_column_path_units(pred_used["columns"], column_graph)

        metrics = compute_metrics(
            gold_used,
            pred_used,
            gold_table_paths,
            pred_table_paths,
            gold_column_paths,
            pred_column_paths,
        )
        subgraph_metrics = compute_subgraph_metrics(gold_used, pred_used, table_graph)

        results.append(
            {
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "gold_tables": sorted(gold_used["tables"]),
                "pred_tables": sorted(pred_used["tables"]),
                "gold_columns": sorted(gold_used["columns"]),
                "pred_columns": sorted(pred_used["columns"]),
                "gold_kg_table_paths": sorted(gold_table_paths),
                "pred_kg_table_paths": sorted(pred_table_paths),
                "gold_kg_column_paths": sorted(gold_column_paths),
                "pred_kg_column_paths": sorted(pred_column_paths),
                **metrics,
                **subgraph_metrics,
            }
        )

    summary = {}
    if results:
        metric_keys = [k for k, v in results[0].items() if isinstance(v, float)]
        for k in metric_keys:
            summary[k] = sum(r[k] for r in results) / len(results)

    return results, summary, parser_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Text-to-SQL outputs with global-KG path and subgraph repair metrics."
    )
    parser.add_argument("--input-glob", default="*_dev_spider_syn.json", help="Input file glob pattern")
    parser.add_argument("--output", default="all_models_evaluation_summary_kg_path.json", help="Output summary JSON file")
    parser.add_argument("--kg-file", default="spider_dev_only_schema_kg.json", help="Global KG JSON file")
    args = parser.parse_args()

    db_to_tables, table_graph_by_db, column_graph_by_db = load_global_kg(Path(args.kg_file))

    input_files = sorted(Path(".").glob(args.input_glob))
    output_path = Path(args.output)

    all_results = []
    for file_path in input_files:
        results, summary, parser_stats = evaluate_file(
            str(file_path),
            db_to_tables,
            table_graph_by_db,
            column_graph_by_db,
        )
        all_results.append(
            {
                "file": file_path.name,
                "num_examples": len(results),
                "summary_metrics": summary,
                "parser_usage": parser_stats,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Wrote consolidated evaluation summary to: {output_path}")
    print(f"Evaluated {len(all_results)} files.")
    for item in all_results:
        print(f"\n{item['file']} ({item['num_examples']} examples)")
        for k, v in item["summary_metrics"].items():
            print(f"{k}: {v:.4f}")
        print("Parser usage:")
        for k, v in item["parser_usage"].items():
            print(f"{k}: {v}")
