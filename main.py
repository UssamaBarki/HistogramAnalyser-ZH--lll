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

# ----------------------------------------------------------------------------------
# CSS Styling - Toggle Buttons
# ----------------------------------------------------------------------------------

# Custom CSS: Toggle buttons default to white, selected to green
pn.config.raw_css.append("""
/* Inactive (unselected) toggle buttons */
.bk-btn-group .bk-btn {
    background-color: white !important;
    color: black !important;
    border: 1px solid black !important;
}

/* Active (selected) toggle buttons */
.bk-btn-group .bk-btn.bk-active {
    background-color: #2ca02c !important;
    color: white !important;
}
""")

# ----------------------------------------------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------------------------------------------

def load_and_tag(path: str, label: str) -> pd.DataFrame:
    """
    Load CSV data from a given path and tag it with a 'Process' label.

    Parameters:
        path (str): Path to CSV file.
        label (str): Label to assign to 'Process' column.

    Returns:
        pd.DataFrame: Tagged DataFrame.
    """
    df = pd.read_csv(path)
    df["Process"] = label
    return df

# Load and concatenate all datasets with respective process labels
df = pd.concat([
    load_and_tag("Data/DM_200.csv", "Z + Dark Matter"),
    load_and_tag("Data/ZZ.csv", "ZZ"),
    load_and_tag("Data/WZ.csv", "WZ"),
    load_and_tag("Data/Z+jets.csv", "Zjets"),
    load_and_tag("Data/Non-resonant_ll.csv", "Non-resonant ll")
], ignore_index=True).dropna()

# ---------------------------------------------------------------
# Configuration for histogram plotting and process colors
# ---------------------------------------------------------------

# Define columns to plot with default cut ranges
PLOT_COLUMNS = {
    "Mll [GeV]": ("mll", (df["mll"].min(), df["mll"].max())),
    "MET [GeV]": ("ETmiss", (df["ETmiss"].min(), df["ETmiss"].max())),
    "MET/HT (sig.)": ("ETmiss_over_HT", (df["ETmiss_over_HT"].min(), df["ETmiss_over_HT"].max())),
    "ΔR(ll)": ("dRll", (df["dRll"].min(), df["dRll"].max())),
    "Δϕ(ll,MET)": ("dphi_pTll_ETmiss", (df["dphi_pTll_ETmiss"].min(), df["dphi_pTll_ETmiss"].max())),
    "Frac. pₜ diff": ("fractional_pT_difference", (df["fractional_pT_difference"].min(), df["fractional_pT_difference"].max())),
    "Lead lep pₜ [GeV]": ("lead_lep_pt", (df["lead_lep_pt"].min(), df["lead_lep_pt"].max())),
    "Sub-lead lep pₜ [GeV]": ("sublead_lep_pt", (df["sublead_lep_pt"].min(), df["sublead_lep_pt"].max())),
    "B-jets": ("N_bjets", (df["N_bjets"].min(), df["N_bjets"].max())),
    "Sum lep charge": ("sum_lep_charge", (df["sum_lep_charge"].min(), df["sum_lep_charge"].max())),
}

NORMAL_COLORS = {
    "Z + Dark Matter": "#d62728",    # Crimson Red
    "ZZ": "#2ca02c",                 # Medium Green
    "WZ": "#ff7f0e",                 # Dark Orange
    "Zjets": "#1e90ff",              # Dodger Blue
    "Non-resonant ll": "#808080"     # Gray
}

COLORBLIND_SAFE_COLORS = {
    "Z + Dark Matter": "#cc79a7",    # Bright Purple (CB safe)
    "ZZ": "#009e73",                 # Bluish Green (CB safe)
    "WZ": "#e69f00",                 # Orange (CB safe)
    "Zjets": "#56b4e9",              # Sky Blue (CB safe)
    "Non-resonant ll": "#999999"     # Gray (CB safe)
}

PROCESS_COLORS = NORMAL_COLORS.copy()  # Default Color

PROCESS_LIST = list(PROCESS_COLORS.keys())

# Precompute total counts for each process (used for percentage bars)
FULL_COUNTS = {proc: int((df.Process == proc).sum()) for proc in PROCESS_LIST}

def make_histogram(data: pd.DataFrame, column: str, edges: np.ndarray) -> np.ndarray:
    """
    Generate a histogram for a specific column and bin edges.

    Parameters:
        data (pd.DataFrame): Subset of the data.
        column (str): Column to histogram.
        edges (np.ndarray): Histogram bin edges.

    Returns:
        np.ndarray: Bin counts.
    """
    weights = data.get("totalWeight", None)
    hist, _ = np.histogram(data[column], bins=edges, weights=weights)
    return hist

# ----------------------------------------------------------------------------------
# Dashboard Class
# ----------------------------------------------------------------------------------
class CrossFilteringHist(param.Parameterized):
    """
    Interactive dashboard class with cross-filtering histograms using Bokeh and Panel.
    Allows selection of processes and filtering on variables with sliders.
    Updates histograms and statistics dynamically based on filters.
    """

    processes = param.List(default=PROCESS_LIST.copy())

    def __init__(self, **params):
        super().__init__(**params)

        # Div element to display event counts and significance info
        self.count_div = Div()
        # Holds BoxAnnotations (shaded regions) for histograms to indicate filtering range
        self.shadows = {}
        self.sources = {}
        self.labels = {}
        self.max_y_seen = {}

        self.cb_toggle = pn.widgets.Toggle(
            name="Color Vision Deficiency (CVD) - Mode",
            value=False,  # Off by default
            button_type='primary'
        )
        self.cb_toggle.param.watch(self._update_color_scheme, 'value')

        # Toggle buttons for processes
        self.proc_sel = pn.widgets.ToggleGroup(
            name="Processes",
            options=self.processes,
            value=self.processes.copy(),
            button_type='success',
            width=400,
            height=31,
            sizing_mode='fixed',
        )

        self.proc_sel.param.watch(self._on_change, 'value')

        # Dictionary to hold widgets (sliders or checkbox groups) keyed by plot title
        self.widgets = {}

        # Dictionary to hold figures and related histogram bin info keyed by plot title
        self.figs = {}

        # Initialize pie chart and histograms
        self._init_pie_chart()
        self._init_histograms()

        # Initial update to sync visuals
        self._on_change()

        # Compose layout
        self.layout = self._make_layout()

    def _update_color_scheme(self, event=None):
        """
        Update color scheme based on toggle value.
        This updates glyph colors in-place for faster responsiveness.
        """
        global PROCESS_COLORS

        # Update color mapping globally
        PROCESS_COLORS = COLORBLIND_SAFE_COLORS.copy() if self.cb_toggle.value else NORMAL_COLORS.copy()

        # Update pie chart colors
        if 'color' in self.pie_source.data:
            new_colors = [PROCESS_COLORS.get(proc, "#cccccc") for proc in self.pie_source.data['process']]
            self.pie_source.data['color'] = new_colors

        # Update histogram bar colors: each stacked layer is a separate glyph
        for title, (fig, edges, mids, width) in self.figs.items():
            # Collect all vbar glyph renderers (each is one stack layer)
            vbars = [r for r in fig.renderers if hasattr(r, 'glyph') and r.glyph.__class__.__name__ == 'VBar']

            # The number of vbars should match number of processes
            for idx, glyph_renderer in enumerate(vbars):
                proc = PROCESS_LIST[idx]
                new_color = PROCESS_COLORS.get(proc, "#000000")
                glyph_renderer.glyph.fill_color = new_color
                glyph_renderer.glyph.line_color = None

        # Trigger update for pie chart + counts + histograms
        self._on_change()


    def _on_change(self, *events):
        """
        Main update function called when any widget or selection changes.
        Updates pie chart, counts display, and histograms.
        """
        mask = self._compute_mask()
        sel = self.proc_sel.value

        # Calculate signal and background counts for significance metric
        signal_count = int(((df.Process == "Z + Dark Matter") & mask).sum())
        background_count = int((df.Process.isin([k for k in PROCESS_COLORS if k != "Z + Dark Matter"]) & mask).sum())
        significance = signal_count / np.sqrt(background_count) if background_count else 0
        significance_pct = int(min(significance, 5) / 5 * 100)

        self._update_pie_chart(mask, sel)
        self._update_count_div(mask, significance, significance_pct)
        self._update_histograms(mask, sel, events)


    def _init_pie_chart(self):
        """
        Initializes pie chart to show process contributions.
        """
        self.pie_source = ColumnDataSource(dict(process=[], value=[], angle=[], color=[]))
        self.pie = figure(
            height=245, width=245,
            toolbar_location=None,
            tools="hover",
            tooltips="@process: @value",
            title="Share In Total Events"
        )
        self.pie.title.align = "center"
        self.pie.title.text_font_style = "bold"
        self.pie.toolbar.logo = None
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
        """
        Initializes histogram figures and their corresponding filter widgets.
        """
        # Columns with known negative values in Zjets to consider for slider min range
        cols_allow_neg = ["lead_lep_pt", "mll", "fractional_pT_difference", "ETmiss_over_HT"]

        for title, (col, _) in PLOT_COLUMNS.items():
            if col == "sum_lep_charge":
                #  Categorical toggle group for Sum Lepton Charge
                edges = np.array([-3, -1, 1, 3])
                mids = np.array([-2, 0, 2])
                width = 2

                widget = pn.widgets.ToggleGroup(
                    name=title,
                    options=[-2, 0, 2],
                    value=[-2, 0, 2],
                    button_type='success',
                    width=200,
                    height=31,
                    sizing_mode='fixed',
                    margin=(0, 0, 0, 50),
                    align='center',
                )

            elif col == "N_bjets":
                # Categorical toggle group for BTags
                edges = np.array([-0.5, 0.5, 1.5])
                mids = np.array([0, 1])
                width = 0.95

                widget = pn.widgets.ToggleGroup(
                    name=title,
                    options=[0, 1],
                    value=[0, 1],
                    button_type='success',
                    width=120,
                    height=31,
                    sizing_mode='fixed',
                    margin=(0, 0, 0, 50),
                    align='center',
                )

            else:
                # Continuous slider for other variables
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
                    name=title,
                    start=lo,
                    end=hi,
                    value=(lo, hi),
                    step=(hi - lo) / 100 or 1,
                    width=260,
                    bar_color="lightgrey",
                    format='0.00',
                    margin=(0, 0, 0, 42),
                )
                widget.visible = True

            # Watch for changes to slider or checkbox to update filtering
            widget.param.watch(self._on_change, 'value')

            # Create histogram figure
            fig = figure(title=title, width=300, height=240, tools="reset")
            fig.toolbar_location = None
            fig.toolbar.logo = None
            fig.toolbar.active_drag = None
            fig.xaxis.axis_label = title
            fig.yaxis.axis_label = "Events"
            fig.x_range = Range1d(edges[0], edges[-1])
            fig.y_range = Range1d(0, 1)
            self.max_y_seen[title] = 1

            # For continuous variables add box select tools and shaded filters
            if col not in ["sum_lep_charge", "N_bjets"]:
                fig.add_tools(BoxSelectTool(dimensions="width"))
                cb = self._make_select_cb(title, edges, widget)
                for evt in (SelectionGeometry, DoubleTap, Reset):
                    fig.on_event(evt, cb)

                left_shadow = BoxAnnotation(left=edges[0], right=edges[0], fill_color="lightgrey", fill_alpha=0.3)
                right_shadow = BoxAnnotation(left=edges[-1], right=edges[-1], fill_color="lightgrey", fill_alpha=0.3)
                fig.add_layout(left_shadow)
                fig.add_layout(right_shadow)
                self.shadows[title] = (left_shadow, right_shadow)

            source = ColumnDataSource(data={"x": mids, **{p: np.zeros_like(mids) for p in PROCESS_LIST}})
            self.sources[title] = source

            fig.vbar_stack(
                PROCESS_LIST, x='x', width=width,
                color=[PROCESS_COLORS[p] for p in PROCESS_LIST],
                source=source
            )

            self.widgets[title] = widget
            self.figs[title] = (fig, edges, mids, width)

            # Set custom tickers and labels for sum_lep_charge and BTags
            if col == "sum_lep_charge":
                fig.xaxis.ticker = [-2, 0, 2]
                fig.xaxis.major_label_overrides = {-2: "-2", 0: "0", 2: "2"}

            elif col == "N_bjets":
                fig.xaxis.ticker = [0, 1]
                fig.xaxis.major_label_overrides = {0: "0", 1: "1"}

    def _make_select_cb(self, title, edges, widget):
        """
        Creates a callback for box select and reset events to update the slider widget.
        """
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
        return callback

    def _compute_mask(self) -> pd.Series:
        """
        Compute boolean mask filtering dataframe according to
        selected processes and slider/checkbox widget values.
        """
        mask = df.Process.isin(self.proc_sel.value)
        for title, (col, _) in PLOT_COLUMNS.items():
            widget = self.widgets[title]
            if col == "sum_lep_charge":
                # For categorical, filter exact matches
                mask &= df[col].isin(widget.value)
            elif col == "N_bjets":
                mask &= df[col].isin(widget.value)
            else:
                # For continuous, filter by slider range
                mask &= df[col].between(*widget.value)
        return mask

    def _update_pie_chart(self, mask: pd.Series, selected_processes: list):
        """
        Update pie chart data source according to filtered data.
        """
        counts = {proc: int(((df.Process == proc) & mask).sum()) for proc in selected_processes}
        pie_df = pd.Series(counts).reset_index(name='value').rename(columns={'index': 'process'})
        pie_df = pie_df[pie_df.value > 0]
        pie_df['angle'] = pie_df['value'] / pie_df['value'].sum() * 2 * pi
        pie_df['color'] = pie_df['process'].map(PROCESS_COLORS)

        self.pie_source.data = pie_df.to_dict(orient='list')

    def _update_count_div(self, mask: pd.Series, significance: float, significance_pct: int):
        """
        Update HTML div displaying counts and significance metric.
        """
        total_events = int(mask.sum())
        html = [f"<h3>Total Events: {total_events}</h3>"]
        html.append(
            f"<div style='display: flex; justify-content: space-between; align-items: center; "
            f"margin-bottom: 4px; font-weight: bold; font-size: 0.95em;'>"
            f"  <div style='width: 160px;'>Events Count</div>"
            f"  <div style='width: 250px; text-align: center; margin-left: 10px;'>Percentage Yield</div>"
            f"</div>"
        )

        for i, proc in enumerate(self.processes):
            count = int(((df.Process == proc) & mask).sum())
            pct = int(count / FULL_COUNTS[proc] * 100) if FULL_COUNTS[proc] else 0

            # Bar with count
            bar = (
                f"<div style='display: flex; align-items: center; margin: 4px 0; width: 100%;'>"
                f"  <span style='width: 1em; color: {PROCESS_COLORS[proc]};'>■</span>"
                f"  <div style='width: 140px; margin-left: 0.5em; font-weight: bold;'>{proc}: {count}</div>"
                f"  <div style='flex-grow: 1; margin-left: 0.5em; position: relative; height: 0.8em; min-width: 200px;'>"
                f"    <div style='width: 100%; height: 100%; background: #f5f5f5; border: 1px solid #aaa;'></div>"
                f"    <div style='position: absolute; top: 0; left: 0; height: 100%; width: {pct}%; background: {PROCESS_COLORS[proc]};'></div>"
                f"  </div>"
                f"  <div style='width: 40px; margin-left: 10px; text-align: left; font-weight: bold;'>{pct}%</div>"  # Percentage label next to bar
                f"</div>"
            )

            html.append(bar)

        sig_bar = (
            f"<div style='display:flex; align-items:center; margin:8px 0 4px; width:100%; flex-direction:column;'>"
            f"  <div style='display:flex; align-items:center; width:100%;'>"
            f"    <span style='width:1em; color: black;'>■</span>"
            f"    <div style='width:140px; margin-left:0.5em; font-weight:bold;'>Significance: {significance:.2f}σ</div>"
            f"    <div style='width: 206px; margin-left: 0.5em;'>"
            f"      <div style='position:relative; height:0.8em; width:100%;'>"
            f"        <div style='width:100%; height:100%; background:#f5f5f5; border:1px solid #aaa;'></div>"
            f"        <div style='position:absolute; top:0; left:0; height:100%; width:{significance_pct}%; background:black;'></div>"
            f"      </div>"

            f"      <div style='display:flex; justify-content:space-between; font-size:0.75em; margin-top:2px; color:#444;'>"
            f"        <span>0σ</span><span>1σ</span><span>2σ</span><span>3σ</span><span>4σ</span><span>5σ</span>"
            f"      </div>"
            f"    </div>"
            f"  </div>"
            f"</div>"
        )

        html.append(sig_bar)

        self.count_div.text = ''.join(html)

    def _update_histograms(self, mask: pd.Series, selected_processes: list, events):
        """
        Update each histogram's bars, axis ranges, and shaded filters based on current selection.
        """
        for title, (col, _) in PLOT_COLUMNS.items():
            fig, edges, mids, width = self.figs[title]
            widget = self.widgets[title]
            source = self.sources[title]

            # Prepare histogram data for selected processes
            data = {"x": mids}
            for proc in PROCESS_LIST:
                if proc in selected_processes:
                    filtered_data = df[(df.Process == proc) & mask]
                    data[proc] = make_histogram(filtered_data, col, edges)
                else:
                    data[proc] = np.zeros_like(mids)
            source.data = data

            # Update shaded filter boxes on histogram if applicable
            if title in self.shadows:
                left_shadow, right_shadow = self.shadows[title]
                left_shadow.right = widget.value[0]
                right_shadow.left = widget.value[1]

            heights = np.sum([data[proc] for proc in selected_processes], axis=0) if selected_processes else []
            if len(heights):
                current_max = heights.max()
                if current_max > self.max_y_seen[title] * 1.05 or current_max < self.max_y_seen[title] * 0.95:
                    fig.y_range.end = max(current_max * 1.1, 1)
                    self.max_y_seen[title] = fig.y_range.end

    def _make_layout(self) -> pn.Column:
        """
        Arrange widgets and figures into a Panel layout.
        First row: 5 histograms, Second row: 4 histograms,
        plus top row with counts, process selector, pie chart, and instruction box.
        """
        rows = []
        titles = list(PLOT_COLUMNS)

        # First row: 5 histograms
        row1 = pn.Row(
            *[pn.Column(self.figs[t][0], self.widgets[t]) for t in titles[:5]],
            sizing_mode="stretch_width"
        )

        # Second row: 4 histograms
        row2 = pn.Row(
            *[pn.Column(self.figs[t][0], self.widgets[t]) for t in titles[5:]],
            sizing_mode="stretch_width"
        )

        # Instruction box (right of pie chart)
        instruction_text = pn.pane.Markdown(
            """
            ### Instructions

            - Use green colored **toggle buttons** to select/deselect processes.
            - Adjust **sliders** to apply range-based filters.
            - **Double-click** a histogram to reset its range.
            - For people with color deficiency, click the **Color Vision Deficiency (CVD) - Mode** toggle button at the top right corner to change the colors.
            """,
            width=600,
            sizing_mode=None,
            margin=(10, 10, 10, 10)
        )

        # Combined title and toggle in one row
        title_row = pn.Row(
            pn.pane.HTML(
                "<h1 style='margin: 0; padding-top: 10px;'>Histogram Analyzer to Find Dark Matter</h1>",
                width=1160
            ),
            pn.Spacer(width=100),
            pn.Column(self.cb_toggle, margin=(10, 0, 0, 0)),
            height=60,
            sizing_mode=None,
            margin=(0, 0, 0, 0)
        )

        # Main row with counts, pie chart, and instructions
        top_row = pn.Row(
            pn.Column(self.count_div, self.proc_sel),
            pn.Column(self.pie, margin=(12, 0, 0, 0)),
            instruction_text,
            sizing_mode="stretch_width"
        )

        return pn.Column(title_row, top_row, row1, row2, sizing_mode="stretch_width")


dashboard = CrossFilteringHist()
dashboard.layout.servable()
