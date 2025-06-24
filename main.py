import numpy as np
import pandas as pd
from math import pi

from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, Div, Range1d, BoxAnnotation,
    BoxSelectTool
)
from bokeh.events import SelectionGeometry, DoubleTap, Reset
from bokeh.transform import cumsum

import panel as pn
import param

pn.extension()


def load_and_tag(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Process"] = label
    return df


df = pd.concat([
    load_and_tag("DM_200.csv", "ZH→inv"),
    load_and_tag("ZZ.csv", "ZZ"),
    load_and_tag("WZ.csv", "WZ"),
    load_and_tag("Z+jets.csv", "Zjets"),
    load_and_tag("Non-resonant_ll.csv", "Non-resonant ℓℓ")
], ignore_index=True).dropna()

df.rename(columns={
    "mll": "Mll",
    "ETmiss": "MET",
    "ETmiss_over_HT": "MET_sig",
    "lead_lep_pt": "lep1_pt",
    "sublead_lep_pt": "lep2_pt",
    "dRll": "deltaR_ll",
    "dphi_pTll_ETmiss": "dphi_ll_met",
    "fractional_pT_difference": "frac_pt_diff",
    "N_bjets": "BTags"
}, inplace=True)

df["Channel"] = "ℓℓ"

PLOT_COLUMNS = {
    "Dilepton mass [GeV]": ("Mll", (76, 106)),
    "Missing ET [GeV]": ("MET", (90, df.MET.max())),
    "MET/HT (sig.)": ("MET_sig", (0.9, df.MET_sig.max())),
    "ΔR(ℓℓ)": ("deltaR_ll", (0, 1.8)),
    "Δϕ(ℓℓ,MET)": ("dphi_ll_met", (2.6, df.dphi_ll_met.max())),
    "Frac. pₜ diff": ("frac_pt_diff", (0, 0.3)),
    "Lead ℓ pₜ [GeV]": ("lep1_pt", (30, df.lep1_pt.max())),
    "Sublead ℓ pₜ [GeV]": ("lep2_pt", (20, df.lep2_pt.max())),
    "B-tags": ("BTags", (0, int(df.BTags.max()))),
    "Sum lep charge": ("sum_lep_charge", None),
}

PROCESS_COLORS = {
    "ZH→inv": "forestgreen",
    "ZZ": "firebrick",
    "WZ": "dodgerblue",
    "Zjets": "goldenrod",
    "Non-resonant ℓℓ": "dimgray"
}

PROCESS_LIST = list(PROCESS_COLORS.keys())
FULL_COUNTS = {proc: int((df.Process == proc).sum()) for proc in PROCESS_LIST}


def make_histogram(data: pd.DataFrame, column: str, edges: np.ndarray) -> np.ndarray:
    weights = data.get("totalWeight", None)
    hist, _ = np.histogram(data[column], bins=edges, weights=weights)
    return hist


class CrossFilteringHist(param.Parameterized):
    processes = param.List(default=PROCESS_LIST.copy())

    def __init__(self, **params):
        super().__init__(**params)

        self.count_div = Div()
        self.shadows = {}

        self.proc_sel = pn.widgets.CheckBoxGroup(
            name="Processes", options=self.processes, value=self.processes.copy(), inline=True
        )
        self.proc_sel.param.watch(self._on_change, 'value')

        self.widgets = {}
        self.figs = {}

        self._init_pie_chart()
        self._init_histograms()
        self._on_change()
        self.layout = self._make_layout()

    def _init_pie_chart(self):
        self.pie_source = ColumnDataSource(dict(process=[], value=[], angle=[], color=[]))
        self.pie = figure(
            height=300, width=300,
            toolbar_location=None,
            tools="hover",
            tooltips="@process: @value"
        )
        self.pie.wedge(
            x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'),
            line_color="white", fill_color='color',
            source=self.pie_source
        )
        self.pie.axis.visible = False
        self.pie.grid.visible = False

    def _init_histograms(self):
        cols_allow_neg = ["lep1_pt", "Mll", "frac_pt_diff", "MET_sig"]

        for title, (col, _) in PLOT_COLUMNS.items():
            if col == "sum_lep_charge":
                edges = np.array([-3, -1, 1, 3])
                mids = np.array([-2, 0, 2])
                width = 2
                widget = pn.widgets.CheckBoxGroup(
                    name=title, options=[-2, 0, 2], value=[-2, 0, 2], inline=True
                )
            else:
                lo, hi = df[col].min(), df[col].max()
                if col in cols_allow_neg:
                    zjets_vals = df.loc[df.Process == "Zjets", col]
                    if len(zjets_vals):
                        zjets_min = zjets_vals.min()
                        if zjets_min < lo and zjets_min < 0:
                            lo = zjets_min
                num_bins = 15
                edges = np.linspace(lo, hi, num_bins + 1)
                mids = (edges[:-1] + edges[1:]) / 2
                width = (edges[1] - edges[0]) * 0.9
                widget = pn.widgets.RangeSlider(
                    name=title, start=lo, end=hi,
                    value=(lo, hi),
                    step=(hi - lo) / 100 or 1,
                    width=300
                )
                widget.visible = False
            widget.param.watch(self._on_change, 'value')

            fig = figure(
                title=title,
                width=300, height=240,
                tools="reset"
            )
            fig.xaxis.axis_label = title
            fig.yaxis.axis_label = "Events"
            fig.x_range = Range1d(edges[0], edges[-1])
            fig.y_range = Range1d(0, 1)

            if col != "sum_lep_charge":
                fig.add_tools(BoxSelectTool(dimensions="width"))
                cb = self._make_select_cb(title, edges, widget)
                for evt in (SelectionGeometry, DoubleTap, Reset):
                    fig.on_event(evt, cb)

                left_shadow = BoxAnnotation(left=edges[0], right=edges[0], fill_color="lightgrey", fill_alpha=0.3)
                right_shadow = BoxAnnotation(left=edges[-1], right=edges[-1], fill_color="lightgrey", fill_alpha=0.3)
                fig.add_layout(left_shadow)
                fig.add_layout(right_shadow)
                self.shadows[title] = (left_shadow, right_shadow)

            self.widgets[title] = widget
            self.figs[title] = (fig, edges, mids, width)

    def _make_select_cb(self, title, edges, widget):
        def callback(event):
            if hasattr(event, 'geometry'):
                lo, hi = sorted((event.geometry["x0"], event.geometry["x1"]))
                new_val = (max(edges[0], lo), min(edges[-1], hi))
                if widget.value != new_val:
                    widget.value = new_val
            else:
                reset_val = (widget.start, widget.end)
                if widget.value != reset_val:
                    widget.value = reset_val
            # No manual self._on_change() here — widget watch handles it
        return callback

    def _compute_mask(self) -> pd.Series:
        mask = df.Process.isin(self.proc_sel.value)
        for title, (col, _) in PLOT_COLUMNS.items():
            widget = self.widgets[title]
            if col == "sum_lep_charge":
                mask &= df[col].isin(widget.value)
            else:
                mask &= df[col].between(*widget.value)
        return mask

    def _on_change(self, *events):
        mask = self._compute_mask()
        sel = self.proc_sel.value

        signal_count = int(((df.Process == "ZH→inv") & mask).sum())
        background_count = int((df.Process.isin([k for k in PROCESS_COLORS if k != "ZH→inv"]) & mask).sum())
        significance = signal_count / np.sqrt(background_count) if background_count else 0
        significance_pct = int(min(significance, 5) / 5 * 100)

        self._update_pie_chart(mask, sel)
        self._update_count_div(mask, significance, significance_pct)
        self._update_histograms(mask, sel, events)

    def _update_pie_chart(self, mask: pd.Series, selected_processes: list):
        counts = {proc: int(((df.Process == proc) & mask).sum()) for proc in selected_processes}
        pie_df = pd.Series(counts).reset_index(name='value').rename(columns={'index': 'process'})
        pie_df = pie_df[pie_df.value > 0]
        pie_df['angle'] = pie_df['value'] / pie_df['value'].sum() * 2 * pi
        pie_df['color'] = pie_df['process'].map(PROCESS_COLORS)

        self.pie_source.data = pie_df.to_dict(orient='list')

    def _update_count_div(self, mask: pd.Series, significance: float, significance_pct: int):
        total_events = int(mask.sum())
        html = [f"<h3>Total Events: {total_events}</h3>"]

        for proc in self.processes:
            count = int(((df.Process == proc) & mask).sum())
            pct = int(count / FULL_COUNTS[proc] * 100) if FULL_COUNTS[proc] else 0
            bar = (
                f"<div style='display:flex;align-items:center;margin:3px 0;'>"
                f"<span style='width:1em;color:{PROCESS_COLORS[proc]};'>■</span>"
                f"<strong style='margin:0 0.5em'>{proc}: {count}</strong>"
                f"<div style='flex:1;height:0.8em;border:1px solid #aaa;background:#f5f5f5;position:relative;'>"
                f"<div style='position:absolute;top:0;right:0;width:{pct}%;height:100%;background:{PROCESS_COLORS[proc]};'></div>"
                f"</div></div>"
            )
            html.append(bar)

        sig_bar = (
            f"<div style='display:flex;align-items:center;margin:8px 0 4px;'>"
            f"<span style='width:1em;'>↯</span>"
            f"<strong style='margin:0 0.5em'>Significance: {significance:.2f}σ</strong>"
            f"<div style='flex:1;height:0.8em;border:1px solid #aaa;background:#f5f5f5;position:relative;'>"
            f"<div style='position:absolute;top:0;right:0;width:{significance_pct}%;height:100%;background:black;'></div>"
            f"</div></div>"
        )
        html.append(sig_bar)

        self.count_div.text = ''.join(html)

    def _update_histograms(self, mask: pd.Series, selected_processes: list, events):
        for title, (col, _) in PLOT_COLUMNS.items():
            fig, edges, mids, width = self.figs[title]
            widget = self.widgets[title]

            data = {"x": mids}
            for proc in selected_processes:
                filtered_data = df[(df.Process == proc) & mask]
                data[proc] = make_histogram(filtered_data, col, edges)

            fig.renderers = []
            if selected_processes:
                fig.vbar_stack(
                    selected_processes, x='x', width=width,
                    color=[PROCESS_COLORS[proc] for proc in selected_processes],
                    source=ColumnDataSource(data)
                )
            else:
                fig.vbar(x=mids, top=np.zeros_like(mids), width=width, alpha=0)

            if title in self.shadows:
                left_shadow, right_shadow = self.shadows[title]
                left_shadow.right = widget.value[0]
                right_shadow.left = widget.value[1]

            if col == "sum_lep_charge":
                fig.x_range.start, fig.x_range.end = edges[0], edges[-1]
                heights = np.sum([data[proc] for proc in selected_processes], axis=0) if selected_processes else []
                ymax = heights.max() if len(heights) else 0
                fig.y_range.start = 0
                fig.y_range.end = max(ymax * 1.1, 1)
                continue

            freeze = False
            if hasattr(widget, 'start') and widget in [e.obj for e in events]:
                freeze = not (
                    np.isclose(widget.value[0], widget.start) and np.isclose(widget.value[1], widget.end)
                )

            if not freeze:
                vals = df[col][mask] if selected_processes else []
                xmin, xmax = (vals.min(), vals.max()) if len(vals) else (edges[0], edges[-1])
                heights = np.sum([data[proc] for proc in selected_processes], axis=0) if selected_processes else []
                ymax = heights.max() if len(heights) else 0
                fig.x_range.start, fig.x_range.end = xmin, xmax
                fig.y_range.start, fig.y_range.end = 0, max(ymax * 1.1, 1)

    def _make_layout(self) -> pn.Column:
        rows = []
        titles = list(PLOT_COLUMNS)

        for i in range(0, len(titles), 2):
            col1 = pn.Column(self.figs[titles[i]][0], self.widgets[titles[i]])
            if i + 1 < len(titles):
                col2 = pn.Column(self.figs[titles[i + 1]][0], self.widgets[titles[i + 1]])
                rows.append(pn.Row(col1, col2))
            else:
                rows.append(col1)

        top_row = pn.Row(
            pn.Column(self.count_div, self.proc_sel),
            pn.Column(self.pie),
            sizing_mode="stretch_width"
        )

        return pn.Column(top_row, *rows, sizing_mode="stretch_width")


def main():
    dashboard = CrossFilteringHist()
    dashboard.layout.servable()
    pn.serve(dashboard.layout, title="Cross Filtering Histogram Dashboard")


if __name__ == "__main__":
    main()
