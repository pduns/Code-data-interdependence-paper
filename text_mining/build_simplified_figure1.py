import argparse
from pathlib import Path
import sys
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_config import DATA_ROOT


DATA_DIR = DATA_ROOT / "processed_data"
FIGURE_DIR = DATA_ROOT / "visualizations"
SUPPLEMENTARY_FILE = DATA_ROOT / "interdependence_paper_supplementary_data_new.xlsx"

TREEMAP_FILE = DATA_DIR / "data_holder_resource_treemap_counts.csv"
HOLDER_COUNTS_FILE = DATA_DIR / "top_data_holder_counts.csv"
HOLDER_DISTRIBUTION_FILE = (
    DATA_DIR / "full_corpus_data_holder_type_distribution_eu_adjusted.csv"
)
FIELD_DISTRIBUTION_FILE = DATA_DIR / "full_corpus_primary_research_field_distribution.csv"

SVG_FILE = FIGURE_DIR / "paper_figure_data_sources_overview.svg"
PDF_FILE = FIGURE_DIR / "Fig1.pdf"
PREVIEW_FILE = FIGURE_DIR / "paper_figure_data_sources_overview.png"
PANEL_A_SVG_FILE = FIGURE_DIR / "paper_figure_panel_a_simplified.svg"
PANEL_A_PDF_FILE = FIGURE_DIR / "Fig1a.pdf"
PANEL_A_PREVIEW_FILE = FIGURE_DIR / "paper_figure_panel_a_simplified.png"
PANEL_B_PDF_FILE = FIGURE_DIR / "Fig1b.pdf"
PANEL_C_PDF_FILE = FIGURE_DIR / "Fig1c.pdf"

HOLDER_TYPE_COLORS = {
    "national public": "#2f6f9f",
    "international public": "#7a4f9a",
    "research": "#3f8f5f",
    "private": "#b84a4a",
    "ngo-led": "#c27c2c",
    "hybrid": "#8a8a8a",
    "not specified": "#bdbdbd",
}
HOLDER_TYPE_DISPLAY = {
    "national public": "National public",
    "international public": "International public",
    "research": "Research",
    "private": "Private",
    "ngo-led": "NGO-led",
    "hybrid": "Hybrid",
    "not specified": "Not specified",
}
FIELD_COLORS = {
    "Ecology, Biodiversity & Conservation": "#009e73",
    "Earth, Climate, Ocean & Water": "#56b4e9",
    "Agriculture, Forestry & Food": "#e69f00",
    "Environmental Health, Pollution & Risk": "#d55e00",
    "Social, Economic & Policy": "#cc79a7",
}
FIELD_DISPLAY = {
    "Ecology, Biodiversity & Conservation": "Ecology & Conservation",
}
HOLDER_NAME_DISPLAY = {
    "UniProt Consortium": "UniProt",
}


def normalize_sizes(sizes, dx, dy):
    total = sum(sizes)
    return [size * dx * dy / total for size in sizes]


def worst_ratio(row, side):
    if not row:
        return float("inf")
    total = sum(row)
    return max(
        side * side * max(row) / (total * total),
        total * total / (side * side * min(row)),
    )


def layout_row(row, x, y, dx, dy):
    width = sum(row) / dy
    rectangles = []
    current_y = y
    for area in row:
        height = area / width
        rectangles.append((x, current_y, width, height))
        current_y += height
    return rectangles, (x + width, y, dx - width, dy)


def layout_column(row, x, y, dx, dy):
    height = sum(row) / dx
    rectangles = []
    current_x = x
    for area in row:
        width = area / height
        rectangles.append((current_x, y, width, height))
        current_x += width
    return rectangles, (x, y + height, dx, dy - height)


def squarify(sizes, x, y, dx, dy):
    remaining = list(normalize_sizes(sizes, dx, dy))
    rectangles = []
    while remaining:
        side = min(dx, dy)
        row = [remaining.pop(0)]
        while remaining and worst_ratio(row + [remaining[0]], side) <= worst_ratio(row, side):
            row.append(remaining.pop(0))
        if dx >= dy:
            laid_out, (x, y, dx, dy) = layout_row(row, x, y, dx, dy)
        else:
            laid_out, (x, y, dx, dy) = layout_column(row, x, y, dx, dy)
        rectangles.extend(laid_out)
    return rectangles


RESOURCE_FONT_SIZE = 11.2
HOLDER_FONT_SIZE = 12.8
NARROW_HOLDER_FONT_SIZE = 11.4


def apply_eu_public_equivalent_rule(treemap_data):
    """Apply the later-figure EU sensitivity rule to panel-A holder types."""
    eu_holder_flags = pd.read_excel(
        SUPPLEMENTARY_FILE,
        sheet_name="unique_institutions",
        usecols=["data_holder", "EU"],
    )
    eu_holder_flags["data_holder"] = (
        eu_holder_flags["data_holder"].astype("string").str.strip()
    )
    eu_holder_flags["EU"] = pd.to_numeric(
        eu_holder_flags["EU"], errors="coerce"
    ).fillna(0)
    eu_holder_names = set(
        eu_holder_flags.loc[
            eu_holder_flags["EU"].eq(1), "data_holder"
        ].dropna()
    )

    adjusted = treemap_data.copy()
    reclassified = (
        adjusted["standardized_data_holder_name"].isin(eu_holder_names)
        & adjusted["standardized_data_holder_type"].eq("international public")
    )
    changed_holders = sorted(
        adjusted.loc[reclassified, "standardized_data_holder_name"].unique()
    )
    adjusted.loc[reclassified, "standardized_data_holder_type"] = "national public"
    return adjusted, changed_holders


def resource_label(item, width, height, force=False):
    """Return a complete label only when the resource box can contain it."""
    max_characters = max(7, int(width * 1.30))
    resource_name = str(item["name"])
    if " " not in resource_name and len(resource_name) <= 10:
        name_lines = [resource_name]
    else:
        name_lines = wrap(resource_name, width=max_characters)
    fits = width >= 7.0 and height >= 6.3 and len(name_lines) <= 2
    if not fits and not force:
        return None
    name_text = "\n".join(name_lines[:2])
    return f"{name_text}\n{int(item['value']):,} papers"


def consolidate_resources(holder_rows, x, y, width, height, detail_threshold):
    """Use either a labelled Other box or unlabelled tiny boxes, never both."""
    original_items = [
        {
            "name": row["data_resource_for_visualization"],
            "value": float(row["unique_papers"]),
        }
        for _, row in holder_rows.iterrows()
    ]

    small_total = sum(
        item["value"] for item in original_items if item["value"] < detail_threshold
    )
    items = [
        item for item in original_items if item["value"] >= detail_threshold
    ]
    if small_total:
        items.append({"name": "Other resources", "value": small_total})

    while True:
        items = sorted(items, key=lambda item: item["value"], reverse=True)
        rectangles = squarify(
            [item["value"] for item in items], x, y, width, height
        )
        largest_named_index = next(
            (
                index
                for index, item in enumerate(items)
                if item["name"] != "Other resources"
            ),
            None,
        )
        labels = []
        for index, (item, rectangle) in enumerate(zip(items, rectangles)):
            labels.append(
                resource_label(
                    item,
                    rectangle[2],
                    rectangle[3],
                    force=index == largest_named_index,
                )
            )

        empty_named_indices = [
            index
            for index, (item, label) in enumerate(zip(items, labels))
            if item["name"] != "Other resources" and label is None
        ]
        if empty_named_indices:
            other_total = sum(
                item["value"] for item in items if item["name"] == "Other resources"
            )
            other_total += sum(items[index]["value"] for index in empty_named_indices)
            items = [
                item
                for index, item in enumerate(items)
                if index not in empty_named_indices
                and item["name"] != "Other resources"
            ]
            items.append({"name": "Other resources", "value": other_total})
            continue

        other_index = next(
            (
                index
                for index, item in enumerate(items)
                if item["name"] == "Other resources"
            ),
            None,
        )
        if other_index is None or labels[other_index] is not None:
            return items, rectangles, labels

        # If an Other box is too small to label at the larger type size, restore
        # the individual tiny boxes and leave those blank instead. This keeps the
        # two visual conventions mutually exclusive within every holder.
        fallback_items = sorted(
            original_items, key=lambda item: item["value"], reverse=True
        )
        fallback_rectangles = squarify(
            [item["value"] for item in fallback_items], x, y, width, height
        )
        fallback_labels = [
            resource_label(
                item,
                rectangle[2],
                rectangle[3],
                force=index == 0,
            )
            for index, (item, rectangle) in enumerate(
                zip(fallback_items, fallback_rectangles)
            )
        ]
        return fallback_items, fallback_rectangles, fallback_labels


def add_treemap(axis, treemap_data, holder_counts):
    global_total = treemap_data["unique_papers"].sum()
    detail_threshold = global_total * 0.004
    holder_summary = (
        treemap_data.groupby(
            [
                "standardized_data_holder_name",
                "data_holder_for_visualization",
                "standardized_data_holder_type",
            ],
            as_index=False,
        )["unique_papers"]
        .sum()
        .sort_values("unique_papers", ascending=False)
        .reset_index(drop=True)
    )
    holder_rectangles = squarify(holder_summary["unique_papers"].tolist(), 0, 0, 100, 100)
    paper_counts = holder_counts.set_index("standardized_name")["unique_papers"].to_dict()
    simplified_records = []

    for (_, holder), (x, y, width, height) in zip(holder_summary.iterrows(), holder_rectangles):
        y = 100 - y - height
        holder_name = holder["standardized_data_holder_name"]
        display_name = HOLDER_NAME_DISPLAY.get(
            holder_name, holder["data_holder_for_visualization"]
        )
        holder_type = holder["standardized_data_holder_type"]
        color = HOLDER_TYPE_COLORS.get(holder_type, "#bdbdbd")

        axis.add_patch(
            Rectangle(
                (x, y), width, height,
                facecolor=color, edgecolor="white", linewidth=2.0,
                clip_on=False,
            )
        )

        # Every holder receives the same header band. This removes the oversized
        # NCBI gap created by scaling the band with holder height.
        header_height = min(4.2, height * 0.32)
        resource_height = max(height - header_height, 0)
        holder_rows = (
            treemap_data.loc[
                treemap_data["standardized_data_holder_name"].eq(holder_name)
            ]
            .sort_values("unique_papers", ascending=False)
            .reset_index(drop=True)
        )

        if resource_height > 0 and not holder_rows.empty:
            resources, resource_rectangles, resource_labels = consolidate_resources(
                holder_rows,
                x + 0.55,
                y + 0.55,
                max(width - 1.10, 0.01),
                max(resource_height - 0.55, 0.01),
                detail_threshold,
            )
            has_other = any(
                resource["name"] == "Other resources" for resource in resources
            )
            has_empty = any(label is None for label in resource_labels)
            assert not (has_other and has_empty)
            for resource, label, (rx, ry, rw, rh) in zip(
                resources, resource_labels, resource_rectangles
            ):
                axis.add_patch(
                    Rectangle(
                        (rx, ry), rw, rh,
                        facecolor=color, edgecolor="white", linewidth=0.9,
                        clip_on=False,
                    )
                )
                if label is not None:
                    resource_font_size = (
                        RESOURCE_FONT_SIZE if rw >= 7 else 10.0
                    )
                    axis.text(
                        rx + 0.42,
                        ry + rh - 0.68,
                        label,
                        ha="left",
                        va="top",
                        fontsize=resource_font_size,
                        color="white",
                        linespacing=1.18,
                        clip_on=False,
                    )
                simplified_records.append(
                    {
                        "standardized_data_holder_name": holder_name,
                        "data_holder_for_visualization": display_name,
                        "standardized_data_holder_type": holder_type,
                        "data_resource_for_visualization": resource["name"],
                        "unique_papers": resource["value"],
                    }
                )

        count = paper_counts.get(holder_name)
        header = display_name if pd.isna(count) else f"{display_name} · {int(count):,} papers"
        header_width = max(8, int(width * 1.75))
        header_text = "\n".join(wrap(header, width=header_width)[:2])
        holder_font_size = (
            HOLDER_FONT_SIZE if width >= 9 else NARROW_HOLDER_FONT_SIZE
        )
        if len(header) > width * 1.55:
            holder_font_size = min(holder_font_size, 10.8)
        axis.text(
            x + 0.55,
            y + height - 0.72,
            header_text,
            ha="left",
            va="top",
            fontsize=holder_font_size,
            fontweight="bold",
            color="white",
            linespacing=1.05,
            clip_on=False,
        )

    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.set_axis_off()
    return pd.DataFrame(simplified_records)


def add_distribution_bar(axis, data, category, colors, display_map=None, wrap_width=15):
    data = data.sort_values("holder_mentions", ascending=False).reset_index(drop=True)
    total = data["holder_mentions"].sum()
    positions = np.arange(len(data))
    labels = [
        "\n".join(wrap(str((display_map or {}).get(value, value)), width=wrap_width))
        for value in data[category]
    ]
    bars = axis.bar(
        positions,
        data["holder_mentions"],
        width=0.70,
        color=[colors.get(value, "#bdbdbd") for value in data[category]],
        edgecolor="none",
    )
    axis.bar_label(
        bars,
        labels=[
            f"{int(value):,}\n({value / total:.1%})"
            for value in data["holder_mentions"]
        ],
        padding=4,
        fontsize=9.8,
    )
    axis.set_xticks(positions, labels)
    axis.tick_params(axis="x", labelsize=11.25, length=0)
    axis.tick_params(axis="y", labelsize=11.25)
    axis.set_ylabel("Data use instances", fontsize=11.25)
    axis.set_ylim(0, data["holder_mentions"].max() * 1.20)
    axis.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)


def holder_legend(treemap_data):
    holder_types = [
        holder_type
        for holder_type in HOLDER_TYPE_COLORS
        if holder_type in set(treemap_data["standardized_data_holder_type"])
    ]
    return [
        Patch(
            facecolor=HOLDER_TYPE_COLORS[holder_type],
            edgecolor="none",
            label=HOLDER_TYPE_DISPLAY[holder_type],
        )
        for holder_type in holder_types
    ]


def build_panel_a(treemap_data, holder_counts):
    """Create the standalone panel-A review figure without rebuilding panels b/c."""
    figure = plt.figure(figsize=(15, 9.4), facecolor="white")
    treemap_axis = figure.add_axes([0.048, 0.145, 0.93, 0.735])
    simplified = add_treemap(treemap_axis, treemap_data, holder_counts)

    figure.text(0.045, 0.945, "a", fontsize=20, fontweight="bold", va="bottom")
    figure.text(
        0.078,
        0.945,
        "15 most frequently used data holders and their resources",
        fontsize=15,
        va="bottom",
    )
    figure.text(
        0.34,
        0.072,
        "Data holder type",
        fontsize=12.5,
        ha="right",
        va="center",
        color="#333333",
    )
    legend_handles = holder_legend(treemap_data)
    figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.36, 0.072),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=12.5,
        handlelength=0.9,
        handleheight=0.9,
        columnspacing=1.35,
        borderaxespad=0,
    )

    figure.savefig(PANEL_A_SVG_FILE, format="svg", facecolor="white")
    figure.savefig(PANEL_A_PDF_FILE, format="pdf", facecolor="white")
    figure.savefig(PANEL_A_PREVIEW_FILE, format="png", dpi=200, facecolor="white")
    plt.close(figure)

    print(f"Original resource boxes: {len(treemap_data)}")
    print(f"Panel-A boxes after label-aware consolidation: {len(simplified)}")
    print(f"Saved: {PANEL_A_SVG_FILE}")
    print(f"Saved: {PANEL_A_PDF_FILE}")
    print(f"Saved: {PANEL_A_PREVIEW_FILE}")


def build_distribution_panel(
    data,
    category,
    colors,
    panel_letter,
    title,
    output_file,
    display_map=None,
    wrap_width=15,
    figure_width=8.5,
):
    """Export one lower Figure-1 panel as a standalone vector PDF."""
    figure = plt.figure(figsize=(figure_width, 6.5), facecolor="white")
    axis = figure.add_axes([0.12, 0.16, 0.84, 0.70])
    add_distribution_bar(
        axis,
        data,
        category,
        colors,
        display_map=display_map,
        wrap_width=wrap_width,
    )
    figure.text(
        0.045, 0.93, panel_letter, fontsize=21, fontweight="bold", va="bottom"
    )
    figure.text(0.115, 0.93, title, fontsize=15.5, va="bottom")
    figure.savefig(output_file, format="pdf", facecolor="white")
    plt.close(figure)
    print(f"Saved: {output_file}")


def main(panel_a_only=False):
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            # Keep text as ordinary SVG text. This is substantially lighter than
            # thousands of glyph paths and is handled reliably by Word.
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    treemap_data = pd.read_csv(TREEMAP_FILE)
    treemap_data, eu_reclassified_holders = apply_eu_public_equivalent_rule(
        treemap_data
    )
    holder_counts = pd.read_csv(HOLDER_COUNTS_FILE)
    print(
        "EU holders shown as national public: "
        + ", ".join(eu_reclassified_holders)
    )
    if panel_a_only:
        build_panel_a(treemap_data, holder_counts)
        return

    holder_distribution = pd.read_csv(HOLDER_DISTRIBUTION_FILE)
    field_distribution = pd.read_csv(FIELD_DISTRIBUTION_FILE)

    # Export the three main-text panels individually as vector PDFs before
    # composing the combined figure.
    build_panel_a(treemap_data, holder_counts)
    build_distribution_panel(
        holder_distribution,
        "standardized_data_holder_type",
        HOLDER_TYPE_COLORS,
        "b",
        "Distribution of data uses across holder types",
        PANEL_B_PDF_FILE,
        display_map=HOLDER_TYPE_DISPLAY,
        wrap_width=15,
        figure_width=8.5,
    )
    build_distribution_panel(
        field_distribution,
        "research_field",
        FIELD_COLORS,
        "c",
        "Distribution of data uses across research areas",
        PANEL_C_PDF_FILE,
        display_map=FIELD_DISPLAY,
        wrap_width=17,
        figure_width=9.0,
    )

    figure = plt.figure(figsize=(15, 16), facecolor="white")
    treemap_axis = figure.add_axes([0.048, 0.456, 0.93, 0.485])
    holder_axis = figure.add_axes([0.072, 0.085, 0.405, 0.25])
    field_axis = figure.add_axes([0.555, 0.085, 0.423, 0.25])

    simplified = add_treemap(treemap_axis, treemap_data, holder_counts)
    add_distribution_bar(
        holder_axis,
        holder_distribution,
        "standardized_data_holder_type",
        HOLDER_TYPE_COLORS,
        display_map=HOLDER_TYPE_DISPLAY,
        wrap_width=15,
    )
    add_distribution_bar(
        field_axis,
        field_distribution,
        "research_field",
        FIELD_COLORS,
        display_map=FIELD_DISPLAY,
        wrap_width=17,
    )

    figure.text(0.045, 0.968, "a", fontsize=21, fontweight="bold", va="bottom")
    figure.text(
        0.078,
        0.968,
        "15 most frequently used data holders and their resources",
        fontsize=15.5,
        va="bottom",
    )
    figure.text(0.045, 0.365, "b", fontsize=21, fontweight="bold", va="bottom")
    figure.text(
        0.078,
        0.365,
        "Distribution of data uses across holder types",
        fontsize=15.5,
        va="bottom",
    )
    figure.text(0.525, 0.365, "c", fontsize=21, fontweight="bold", va="bottom")
    figure.text(
        0.558,
        0.365,
        "Distribution of data uses across research areas",
        fontsize=15.5,
        va="bottom",
    )

    legend_handles = holder_legend(treemap_data)
    figure.text(
        0.34,
        0.417,
        "Data holder type",
        fontsize=13.5,
        ha="right",
        va="center",
        color="#333333",
    )
    figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.36, 0.417),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=13.5,
        handlelength=0.9,
        handleheight=0.9,
        columnspacing=1.35,
        borderaxespad=0,
    )

    figure.savefig(SVG_FILE, format="svg", facecolor="white")
    figure.savefig(PDF_FILE, format="pdf", facecolor="white")
    figure.savefig(PREVIEW_FILE, format="png", dpi=200, facecolor="white")
    plt.close(figure)

    print(f"Original resource boxes: {len(treemap_data)}")
    print(f"Simplified resource boxes: {len(simplified)}")
    print(f"Saved: {SVG_FILE}")
    print(f"Saved: {PDF_FILE}")
    print(f"Saved: {PREVIEW_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a-only", action="store_true")
    arguments, _ = parser.parse_known_args()
    main(panel_a_only=arguments.panel_a_only)
