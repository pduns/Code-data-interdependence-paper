from __future__ import annotations

import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET
from math import sqrt
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_config import DATA_ROOT


PROJECT_ROOT = DATA_ROOT

PROCESSED_PATH = PROJECT_ROOT / "processed_data"

BACKBONE_PREFIX = (
    "llm_affiliation_country_database_bipartite_"
    "coscia_noise_corrected_backbone_z1_96_minw2"
)
BACKBONE_EDGE_FILE = PROCESSED_PATH / f"{BACKBONE_PREFIX}_edges.csv"
BACKBONE_NODE_FILE = PROCESSED_PATH / f"{BACKBONE_PREFIX}_nodes.csv"

COUNTRY_FOLD_PREFIX = f"{BACKBONE_PREFIX}_folded_countries"
DATABASE_FOLD_PREFIX = f"{BACKBONE_PREFIX}_folded_databases_min_shared_countries2"
DATABASE_MIN_SHARED_COUNTRIES = 2
DATABASE_INTERPRETABLE_FOLD_PREFIX = (
    f"{BACKBONE_PREFIX}_folded_databases_min_shared_countries5"
)
DATABASE_INTERPRETABLE_MIN_SHARED_COUNTRIES = 5


def clean_xml_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    return "".join(
        ch
        for ch in text
        if (
            ord(ch) in (0x9, 0xA, 0xD)
            or 0x20 <= ord(ch) <= 0xD7FF
            or 0xE000 <= ord(ch) <= 0xFFFD
            or 0x10000 <= ord(ch) <= 0x10FFFF
        )
    )


def indent_xml(element, level=0):
    indentation = "\n" + "  " * level
    child_indentation = "\n" + "  " * (level + 1)
    children = list(element)

    if children:
        if not element.text or not element.text.strip():
            element.text = child_indentation
        for child in children:
            indent_xml(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indentation

    if level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def gexf_attribute_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "double"
    return "string"


def write_gexf(nodes_df, edges_df, output_path, description, edge_label_columns=None):
    edge_label_columns = edge_label_columns or []

    gexf = ET.Element(
        "gexf",
        {
            "xmlns": "http://www.gexf.net/1.2draft",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": (
                "http://www.gexf.net/1.2draft "
                "http://www.gexf.net/1.2draft/gexf.xsd"
            ),
            "version": "1.2",
        },
    )
    meta = ET.SubElement(
        gexf,
        "meta",
        {"lastmodifieddate": pd.Timestamp.today().strftime("%Y-%m-%d")},
    )
    ET.SubElement(meta, "creator").text = "create_backbone_folded_networks.py"
    ET.SubElement(meta, "description").text = description

    graph = ET.SubElement(
        gexf,
        "graph",
        {"mode": "static", "defaultedgetype": "undirected"},
    )

    node_attribute_columns = [
        column
        for column in nodes_df.columns
        if column not in {"node_id", "label"}
    ]
    edge_attribute_columns = [
        column
        for column in edges_df.columns
        if column not in {"source", "target", "weight"}
    ]

    node_attributes = ET.SubElement(
        graph,
        "attributes",
        {"class": "node", "mode": "static"},
    )
    node_attribute_ids = {}
    for idx, column in enumerate(node_attribute_columns):
        attr_id = str(idx)
        node_attribute_ids[column] = attr_id
        ET.SubElement(
            node_attributes,
            "attribute",
            {"id": attr_id, "title": column, "type": gexf_attribute_type(nodes_df[column])},
        )

    edge_attributes = ET.SubElement(
        graph,
        "attributes",
        {"class": "edge", "mode": "static"},
    )
    edge_attribute_ids = {}
    for idx, column in enumerate(edge_attribute_columns, start=len(node_attribute_columns)):
        attr_id = str(idx)
        edge_attribute_ids[column] = attr_id
        ET.SubElement(
            edge_attributes,
            "attribute",
            {"id": attr_id, "title": column, "type": gexf_attribute_type(edges_df[column])},
        )

    nodes_element = ET.SubElement(graph, "nodes")
    for row in nodes_df.itertuples(index=False):
        node_element = ET.SubElement(
            nodes_element,
            "node",
            {"id": clean_xml_text(row.node_id), "label": clean_xml_text(row.label)},
        )
        attvalues = ET.SubElement(node_element, "attvalues")
        for column, attr_id in node_attribute_ids.items():
            ET.SubElement(
                attvalues,
                "attvalue",
                {"for": attr_id, "value": clean_xml_text(getattr(row, column))},
            )

    edges_element = ET.SubElement(graph, "edges")
    for edge_id, row in enumerate(edges_df.itertuples(index=False)):
        label_values = [
            clean_xml_text(getattr(row, column))
            for column in edge_label_columns
            if hasattr(row, column)
        ]
        edge_attributes_map = {
            "id": str(edge_id),
            "source": clean_xml_text(row.source),
            "target": clean_xml_text(row.target),
            "weight": clean_xml_text(row.weight),
        }
        if label_values:
            edge_attributes_map["label"] = " - ".join(label_values)

        edge_element = ET.SubElement(edges_element, "edge", edge_attributes_map)
        attvalues = ET.SubElement(edge_element, "attvalues")
        for column, attr_id in edge_attribute_ids.items():
            ET.SubElement(
                attvalues,
                "attvalue",
                {"for": attr_id, "value": clean_xml_text(getattr(row, column))},
            )

    indent_xml(gexf)
    ET.ElementTree(gexf).write(output_path, encoding="utf-8", xml_declaration=True)


def add_projection_metrics(edge_table, source_activity, target_activity):
    source_counts = edge_table["source"].map(source_activity["degree"]).astype(float)
    target_counts = edge_table["target"].map(target_activity["degree"]).astype(float)
    source_norms = edge_table["source"].map(source_activity["weighted_norm"]).astype(float)
    target_norms = edge_table["target"].map(target_activity["weighted_norm"]).astype(float)

    edge_table["binary_jaccard"] = edge_table["shared_count"] / (
        source_counts + target_counts - edge_table["shared_count"]
    )
    edge_table["binary_cosine"] = edge_table["shared_count"] / (
        source_counts * target_counts
    ).pow(0.5)
    edge_table["weighted_cosine"] = edge_table["weighted_dot_product"] / (
        source_norms * target_norms
    )
    edge_table["binary_overlap_coefficient"] = edge_table["shared_count"] / pd.concat(
        [source_counts, target_counts], axis=1
    ).min(axis=1)
    return edge_table


def build_country_fold(edges, nodes):
    country_edges = edges[
        ["source", "target", "country", "database", "entity_type", "weight"]
    ].copy()
    country_edges["source_name_key"] = (
        country_edges["target"].astype(str)
        + "::"
        + country_edges["entity_type"].astype(str)
        + "::"
        + country_edges["database"].astype(str)
    )

    country_activity = (
        country_edges.groupby("source")
        .agg(
            degree=("source_name_key", "nunique"),
            total_weight=("weight", "sum"),
            weighted_norm=("weight", lambda values: sqrt(float((values**2).sum()))),
        )
    )

    joined = country_edges.merge(
        country_edges,
        on="source_name_key",
        suffixes=("_a", "_b"),
    )
    joined = joined.loc[joined["source_a"].lt(joined["source_b"])].copy()
    joined["min_weight"] = joined[["weight_a", "weight_b"]].min(axis=1)
    joined["dot_product"] = joined["weight_a"].astype(float) * joined["weight_b"].astype(float)

    edge_table = (
        joined.groupby(["source_a", "source_b", "country_a", "country_b"], as_index=False)
        .agg(
            shared_count=("source_name_key", "nunique"),
            shared_weight_min_sum=("min_weight", "sum"),
            weighted_dot_product=("dot_product", "sum"),
        )
        .rename(
            columns={
                "source_a": "source",
                "source_b": "target",
            }
        )
    )
    edge_table.insert(4, "weight", edge_table["shared_count"].astype(int))
    edge_table = add_projection_metrics(edge_table, country_activity, country_activity)
    edge_table = edge_table.sort_values(
        ["shared_count", "shared_weight_min_sum", "weighted_cosine", "country_a", "country_b"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    referenced_nodes = set(edge_table["source"]).union(set(edge_table["target"]))
    node_table = nodes.loc[nodes["node_id"].isin(referenced_nodes)].copy()
    node_table = node_table.merge(
        country_activity.rename(
            columns={
                "degree": "retained_source_count",
                "total_weight": "retained_source_weight",
                "weighted_norm": "retained_source_weighted_norm",
            }
        ),
        left_on="node_id",
        right_index=True,
        how="left",
    )
    return node_table, edge_table.rename(columns={"shared_count": "shared_source_count"})


def create_sqlite_edges(edges, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -400000")

    sql_edges = edges[
        ["source", "target", "country", "database", "entity_type", "weight"]
    ].copy()
    sql_edges.to_sql("edges", conn, index=False, if_exists="replace")
    conn.execute("CREATE INDEX idx_edges_country ON edges(country)")
    conn.execute("CREATE INDEX idx_edges_target ON edges(target)")
    return conn


def read_database_fold_edges(conn):
    query = f"""
        SELECT
            a.target AS source,
            b.target AS target,
            a.database AS database_a,
            b.database AS database_b,
            a.entity_type AS entity_type_a,
            b.entity_type AS entity_type_b,
            COUNT(*) AS shared_country_count,
            SUM(MIN(a.weight, b.weight)) AS shared_weight_min_sum,
            SUM(1.0 * a.weight * b.weight) AS weighted_dot_product
        FROM edges AS a
        JOIN edges AS b
            ON a.country = b.country
           AND a.target < b.target
        GROUP BY
            a.target,
            b.target,
            a.database,
            b.database,
            a.entity_type,
            b.entity_type
        HAVING COUNT(*) >= {DATABASE_MIN_SHARED_COUNTRIES}
        ORDER BY shared_country_count DESC, shared_weight_min_sum DESC
    """
    return pd.read_sql_query(query, conn)


def build_database_fold(edges, nodes):
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "database_fold.sqlite"
        conn = create_sqlite_edges(edges, db_path)
        try:
            edge_table = read_database_fold_edges(conn)
        finally:
            conn.close()

    source_activity = (
        edges.groupby("target")
        .agg(
            degree=("country", "nunique"),
            total_weight=("weight", "sum"),
            weighted_norm=("weight", lambda values: sqrt(float((values**2).sum()))),
        )
    )

    edge_table.insert(6, "weight", edge_table["shared_country_count"].astype(int))
    edge_table = edge_table.rename(columns={"shared_country_count": "shared_count"})
    edge_table = add_projection_metrics(edge_table, source_activity, source_activity)
    edge_table = edge_table.rename(columns={"shared_count": "shared_country_count"})
    edge_table = edge_table.sort_values(
        ["shared_country_count", "shared_weight_min_sum", "weighted_cosine", "database_a", "database_b"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    referenced_nodes = set(edge_table["source"]).union(set(edge_table["target"]))
    node_table = nodes.loc[nodes["node_id"].isin(referenced_nodes)].copy()
    node_table = node_table.merge(
        source_activity.rename(
            columns={
                "degree": "retained_country_count",
                "total_weight": "retained_country_weight",
                "weighted_norm": "retained_country_weighted_norm",
            }
        ),
        left_on="node_id",
        right_index=True,
        how="left",
    )
    return node_table, edge_table


def save_network(nodes_df, edges_df, prefix, description, edge_label_columns):
    node_output = PROCESSED_PATH / f"{prefix}_nodes.csv"
    edge_output = PROCESSED_PATH / f"{prefix}_edges.csv"
    gexf_output = PROCESSED_PATH / f"{prefix}_network.gexf"

    nodes_df.to_csv(node_output, index=False)
    edges_df.to_csv(edge_output, index=False)
    write_gexf(
        nodes_df,
        edges_df,
        gexf_output,
        description=description,
        edge_label_columns=edge_label_columns,
    )

    print(f"Saved nodes: {node_output}")
    print(f"Saved edges: {edge_output}")
    print(f"Saved GEXF: {gexf_output}")
    return node_output, edge_output, gexf_output


def main():
    edges = pd.read_csv(BACKBONE_EDGE_FILE, low_memory=False)
    nodes = pd.read_csv(BACKBONE_NODE_FILE, low_memory=False)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0).astype(int)

    print(f"Loaded {len(edges):,} Coscia-backbone bipartite edges.")
    print(f"Loaded {len(nodes):,} Coscia-backbone bipartite nodes.")

    country_nodes, country_edges = build_country_fold(edges, nodes)
    save_network(
        country_nodes,
        country_edges,
        COUNTRY_FOLD_PREFIX,
        (
            "Country-country projection folded from the Coscia noise-corrected "
            "country-source bipartite backbone. Edge weight is the number of "
            "shared retained source/database nodes."
        ),
        edge_label_columns=["country_a", "country_b"],
    )
    print(
        f"Country fold: {len(country_nodes):,} nodes, {len(country_edges):,} edges; "
        f"max shared sources = {country_edges['shared_source_count'].max():,}."
    )

    database_nodes, database_edges = build_database_fold(edges, nodes)
    save_network(
        database_nodes,
        database_edges,
        DATABASE_FOLD_PREFIX,
        (
            "Database/source-name projection folded from the Coscia noise-corrected "
            "country-source bipartite backbone. Edge weight is the number of shared "
            f"retained countries; edges require at least {DATABASE_MIN_SHARED_COUNTRIES} "
            "shared countries."
        ),
        edge_label_columns=["database_a", "database_b"],
    )
    print(
        f"Database fold: {len(database_nodes):,} nodes, {len(database_edges):,} edges; "
        f"max shared countries = {database_edges['shared_country_count'].max():,}."
    )

    interpretable_database_edges = database_edges.loc[
        database_edges["shared_country_count"].ge(DATABASE_INTERPRETABLE_MIN_SHARED_COUNTRIES)
    ].copy().reset_index(drop=True)
    interpretable_database_node_ids = set(interpretable_database_edges["source"]).union(
        set(interpretable_database_edges["target"])
    )
    interpretable_database_nodes = database_nodes.loc[
        database_nodes["node_id"].isin(interpretable_database_node_ids)
    ].copy().reset_index(drop=True)
    save_network(
        interpretable_database_nodes,
        interpretable_database_edges,
        DATABASE_INTERPRETABLE_FOLD_PREFIX,
        (
            "Stricter database/source-name projection folded from the Coscia "
            "noise-corrected country-source bipartite backbone. Edge weight is "
            "the number of shared retained countries; edges require at least "
            f"{DATABASE_INTERPRETABLE_MIN_SHARED_COUNTRIES} shared countries."
        ),
        edge_label_columns=["database_a", "database_b"],
    )
    print(
        "Interpretable database fold: "
        f"{len(interpretable_database_nodes):,} nodes, "
        f"{len(interpretable_database_edges):,} edges; "
        "max shared countries = "
        f"{interpretable_database_edges['shared_country_count'].max():,}."
    )

    summary = pd.DataFrame(
        [
            {
                "projection": "countries",
                "nodes": len(country_nodes),
                "edges": len(country_edges),
                "edge_weight": "shared_source_count",
                "min_shared_count": 1,
                "max_shared_count": int(country_edges["shared_source_count"].max()),
                "median_shared_count": float(country_edges["shared_source_count"].median()),
            },
            {
                "projection": "databases",
                "nodes": len(database_nodes),
                "edges": len(database_edges),
                "edge_weight": "shared_country_count",
                "min_shared_count": DATABASE_MIN_SHARED_COUNTRIES,
                "max_shared_count": int(database_edges["shared_country_count"].max()),
                "median_shared_count": float(database_edges["shared_country_count"].median()),
            },
            {
                "projection": "databases_interpretable",
                "nodes": len(interpretable_database_nodes),
                "edges": len(interpretable_database_edges),
                "edge_weight": "shared_country_count",
                "min_shared_count": DATABASE_INTERPRETABLE_MIN_SHARED_COUNTRIES,
                "max_shared_count": int(
                    interpretable_database_edges["shared_country_count"].max()
                ),
                "median_shared_count": float(
                    interpretable_database_edges["shared_country_count"].median()
                ),
            },
        ]
    )
    summary_output = PROCESSED_PATH / f"{BACKBONE_PREFIX}_folded_projection_summary.csv"
    summary.to_csv(summary_output, index=False)
    print(f"Saved summary: {summary_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
