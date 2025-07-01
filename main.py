import numpy as np
import pandas as pd
from math import pi

from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, Div, Range1d, BoxAnnotation,
    BoxSelectTool, Label
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
    background-color: green !important;
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
    load_and_tag("DM_200.csv", "ZH→inv"),
    load_and_tag("ZZ.csv", "ZZ"),
    load_and_tag("WZ.csv", "WZ"),
    load_and_tag("Z+jets.csv", "Zjets"),
    load_and_tag("Non-resonant_ll.csv", "Non-resonant ℓℓ")
], ignore_index=True).dropna()

# Standardise column names for convenience
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

# Add a constant channel column (used in analysis/plots)
df["Channel"] = "ℓℓ"

# ---------------------------------------------------------------
# Configuration for histogram plotting and process colors
# ---------------------------------------------------------------

# Define columns to plot with default cut ranges
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


# Colors for each process
PROCESS_COLORS = {
    "ZH→inv": "forestgreen",
    "ZZ": "firebrick",
    "WZ": "dodgerblue",
    "Zjets": "goldenrod",
    "Non-resonant ℓℓ": "dimgray"
}

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

        # Toggle buttons for processes
        self.proc_sel = pn.widgets.ToggleGroup(
            name="Processes",
            options=self.processes,
            value=self.processes.copy(),
            button_type='success',
            width=400,
            height=31,
            sizing_mode='fixed',
            #stylesheets = [{         # Size change for process box
            #"button": {
                #"minWidth": "60px",  # reduce from default ~100px
                #"maxWidth": "60px",  # lock width
                #"padding": "2px 6px",
                #"fontSize": "11px"
            #}
        #}]
        )


        self.proc_sel.param.watch(self._on_change, 'value')

        # Toggle button to show/hide sliders    (Hidden Now)
        #self.slider_toggle = pn.widgets.Toggle(
            #name='Hide / Un-hide Sliders', button_type='primary', value=True
        #)
        #self.slider_toggle.param.watch(self._toggle_sliders, 'value')

        # Dictionary to hold widgets (sliders or checkbox groups) keyed by plot title
        self.widgets = {}

        # Dictionary to hold figures and related histogram bin info keyed by plot title
        self.figs = {}

        # Initialise pie chart and histograms
        self._init_pie_chart()
        self._init_histograms()

        # Initial update to sync visuals
        self._on_change()

        # Compose layout
        self.layout = self._make_layout()


    def _on_change(self, *events):
        """
        Main update function called when any widget or selection changes.
        Updates pie chart, counts display, and histograms.
        """
        mask = self._compute_mask()
        sel = self.proc_sel.value

        # Calculate signal and background counts for significance metric
        signal_count = int(((df.Process == "ZH→inv") & mask).sum())
        background_count = int((df.Process.isin([k for k in PROCESS_COLORS if k != "ZH→inv"]) & mask).sum())
        significance = signal_count / np.sqrt(background_count) if background_count else 0
        significance_pct = int(min(significance, 5) / 5 * 100)

        self._update_pie_chart(mask, sel)
        self._update_count_div(mask, significance, significance_pct)
        self._update_histograms(mask, sel, events)


    def _toggle_sliders(self, event):
        """
        Allows the user to turn on/off sliders (currently unused)
        """
        show = event.new
        for title, widget in self.widgets.items():
            if title == "Sum lep charge":
                # sum lep charge toggle always visible
                continue
            widget.visible = show


    def _init_pie_chart(self):
        """
        Initialises pie chart to show process contributions.
        """
        self.pie_source = ColumnDataSource(dict(process=[], value=[], angle=[], color=[]))
        self.pie = figure(
            height=245, width=245,    # 230 previously
            toolbar_location=None,
            tools="hover",
            tooltips="@process: @value",
            title="Share In Total Events"
        )
        self.pie.title.align = "center"
        #self.pie.title.text_font_size = "19pt"  # or "21pt" to match h3 roughly
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
        Initialises histogram figures and their corresponding filter widgets.
        """
        # Columns with known negative values in Zjets to consider for slider min range
        cols_allow_neg = ["lep1_pt", "Mll", "frac_pt_diff", "MET_sig"]

        for title, (col, _) in PLOT_COLUMNS.items():
            if col == "sum_lep_charge":
                # Categorical variable with fixed bins and checkboxes
                edges = np.array([-3, -1, 1, 3])
                mids = np.array([-2, 0, 2])
                width = 2

                # Centered, spaced toggle buttons for sum_lep_charge
                widget = pn.widgets.ToggleGroup(
                    name=title,
                    options=[-2, 0, 2],
                    value=[-2, 0, 2],
                    button_type='success',
                    width=200,
                    height=31,
                    sizing_mode='fixed',
                    margin=(0,0,0,50), # top, right, bottom, **left**
                    align='center',
                )

            else:
                # Calculate slider range: default min/max from full df
                lo, hi = df[col].min(), df[col].max()

                # If this col is in cols_allow_neg, check Zjets subset for min negative values
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

                # Range slider for continuous variables
                widget = pn.widgets.RangeSlider(
                    name=title,
                    start=lo,
                    end=hi,
                    value=(lo, hi),
                    step=(hi - lo) / 100 or 1,
                    width=260,
                    bar_color="lightgrey",
                    format='0.0',
                    #visible=True,
                    #sizing_mode="fixed", or "stretch_width"
                    #tooltips=True,
                    margin=(0, 0, 0, 42), # top, right, bottom, left
                )
                widget.visible = True
            # Watch for changes to slider or checkbox to update filtering
            widget.param.watch(self._on_change, 'value')

            # Create histogram figure
            fig = figure(title=title, width=300, height=240, tools="reset")
            fig.toolbar_location = None # Turned off bokeh toolbar
            fig.toolbar.logo = None
            fig.toolbar.active_drag = None  # Do not activate BoxSelectTool by default
            fig.xaxis.axis_label = title
            fig.yaxis.axis_label = "Events"
            fig.x_range = Range1d(edges[0], edges[-1])
            fig.y_range = Range1d(0, 1)
            self.max_y_seen[title] = 1

            # For continuous variables add box select tools and shaded filters
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

            source = ColumnDataSource(data={"x": mids, **{p: np.zeros_like(mids) for p in PROCESS_LIST}})
            self.sources[title] = source

            fig.vbar_stack(
                PROCESS_LIST, x='x', width=width,
                color=[PROCESS_COLORS[p] for p in PROCESS_LIST],
                source=source
            )

            self.widgets[title] = widget
            self.figs[title] = (fig, edges, mids, width)

            if col == "sum_lep_charge":
                fig.xaxis.ticker = [-2, 0, 2]
                fig.xaxis.major_label_overrides = {-2: "2", 0: "0", 2: "2"}

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

            # Percentage label above the bar
            #html.append(
                #f"<div style='display:flex;justify-content:flex-end;"
                #f"font-size:0.75em;margin-bottom:0px;color:#444;"
                #f"margin-left:calc(140px + 2em);margin-right:0;'>"
                #f"<span>{pct}%</span></div>"
            #)

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

            # Only under last bar: 0% 50% 100% scale
            #if i == len(self.processes) - 1:
                #html.append(
                    #f"<div style='display:flex;justify-content:space-between;"
                    #f"font-size:0.75em;margin-top:-6px;color:#444;"
                    #f"margin-left:calc(140px + 2em);margin-right:0;'>"
                    #f"<span>0%</span><span>50%</span><span>100%</span></div>"
                #)

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
            f"        <span>0σ</span><span>3σ</span><span>5σ</span>"
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

            - Use **toggle buttons** to select/deselect processes.
            - Adjust **sliders** to apply range-based filters.
            - Double-click a histogram to reset its range. 
            """,
            width=300,  # Fixed width
            sizing_mode=None,  # Avoid using stretch_width with fixed width
            margin=(10, 10, 10, 10)
        )

        # Top row with counts, process selection checkboxes, pie chart, and instructions
        top_row = pn.Row(
            pn.Column(self.count_div, self.proc_sel), #self.slider_toggle),
            #pn.Column(self.pie),
            pn.Column(self.pie, margin=(12, 0, 0, 0)),  # top, right, bottom, left
            instruction_text,
            sizing_mode="stretch_width"
        )

        return pn.Column(top_row, row1, row2, sizing_mode="stretch_width")

def main():
    """
       Entry point to launch the Bokeh Panel server.
       """
    # Instantiate the dashboard and expose its layout for rendering
    dashboard = CrossFilteringHist()
    dashboard.layout.servable()
    pn.serve(dashboard.layout, title="Cross Filtering Histogram Dashboard")

if __name__ == "__main__":
    main()
