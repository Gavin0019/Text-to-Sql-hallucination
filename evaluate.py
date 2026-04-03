import json
import re
from pathlib import Path
from typing import Dict, Set, Tuple, List

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    sqlglot = None
    exp = None


def extract_schema_and_question(input_seq: str) -> Tuple[str, str]:
    """
    Extract:
      - schema DDL text between 'Database Schema:' and 'This schema describes...'
      - question text between 'Question:' and 'Instructions:'
    """
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
    """
    Very simple DDL parser:
    returns {"table_name": {"col1", "col2", ...}, ...}
    """
    schema = {}
    table_blocks = re.findall(
        r"CREATE TABLE\s+[`\"]?(\w+)[`\"]?\s*\((.*?)\);",
        schema_ddl,
        flags=re.DOTALL | re.IGNORECASE,
    )

    for table_name, body in table_blocks:
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
                col_name = m.group(1)
                cols.add(col_name.lower())

        schema[table_name.lower()] = cols

    return schema


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
    """
    Parser-first extractor with sqlglot (fallback to heuristics if unavailable).
    Returns:
      {
        "tables": {"singer"},
        "columns": {"singer.age", "singer.country"}
      }
    """
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

            return {
                "tables": used_tables,
                "columns": used_columns,
            }
        except Exception:
            # Fall back to regex heuristics below if parsing fails.
            if stats is not None:
                stats["fallback_parse_error"] = stats.get("fallback_parse_error", 0) + 1
    else:
        if stats is not None:
            stats["fallback_sqlglot_unavailable"] = stats.get("fallback_sqlglot_unavailable", 0) + 1

    # find tables after FROM / JOIN / UPDATE / INTO
    table_patterns = re.findall(r"\b(?:from|join|update|into)\s+([a-zA-Z_][\w]*)", sql_norm)
    for t in table_patterns:
        if t in schema:
            used_tables.add(t)

    # also fallback: any known table name appearing in SQL
    for table in schema:
        if re.search(rf"\b{re.escape(table)}\b", sql_norm):
            used_tables.add(table)

    # find qualified columns table.column
    qualified_cols = re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", sql_norm)
    for t, c in qualified_cols:
        t = t.lower()
        c = c.lower()
        if t in schema and c in schema[t]:
            used_tables.add(t)
            used_columns.add(f"{t}.{c}")

    # find unqualified columns by matching schema columns
    # only safe if the column name exists in exactly one used table
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

        candidate_tables = []
        for t in used_tables:
            if token in schema[t]:
                candidate_tables.append(t)

        if len(candidate_tables) == 1:
            used_columns.add(f"{candidate_tables[0]}.{token}")

    return {
        "tables": used_tables,
        "columns": used_columns,
    }


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


def compute_metrics(gold_used: Dict[str, Set[str]], pred_used: Dict[str, Set[str]]) -> Dict[str, float]:
    gold_tables = gold_used["tables"]
    pred_tables = pred_used["tables"]
    gold_cols = gold_used["columns"]
    pred_cols = pred_used["columns"]

    gold_schema = gold_tables | gold_cols
    pred_schema = pred_tables | pred_cols

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
    }


def evaluate_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    parser_stats = {
        "sqlglot_used": 0,
        "fallback_parse_error": 0,
        "fallback_sqlglot_unavailable": 0,
    }

    for row in data:
        schema_ddl, question = extract_schema_and_question(row["input_seq"])
        schema = parse_schema_from_ddl(schema_ddl)

        gold_sql = row["output_seq"].strip()
        pred_sql = row["pred_sqls"][0].strip() if row.get("pred_sqls") else ""

        gold_used = extract_used_schema(gold_sql, schema, stats=parser_stats)
        pred_used = extract_used_schema(pred_sql, schema, stats=parser_stats)
        metrics = compute_metrics(gold_used, pred_used)

        results.append({
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "gold_tables": sorted(gold_used["tables"]),
            "pred_tables": sorted(pred_used["tables"]),
            "gold_columns": sorted(gold_used["columns"]),
            "pred_columns": sorted(pred_used["columns"]),
            **metrics
        })

    # average metrics
    summary = {}
    if results:
        metric_keys = [k for k, v in results[0].items() if isinstance(v, float)]
        for k in metric_keys:
            summary[k] = sum(r[k] for r in results) / len(results)

    return results, summary, parser_stats


if __name__ == "__main__":
    input_files = sorted(Path(".").glob("*_dev_spider_syn.json"))
    output_path = Path("all_models_evaluation_summary.json")

    all_results = []
    for file_path in input_files:
        results, summary, parser_stats = evaluate_file(str(file_path))
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
