**Running the Histogram Analyzer:** 

1. Clone the Repository: Download the project to your local machine using an IDE (e.g., PyCharm) or GitHub Desktop.

2. Set Up a Virtual Environment: Create and activate a Python virtual environment within your project directory.

3. Install Dependencies: Use the requirements.txt file to install all necessary packages. In PyCharm: Open the project, navigate to Tools > Sync Python Requirements, and follow the prompt to install the required packages.

4. Run the Application: Execute the main.py file.

5. Launch the Interface: A local server will start, and the histogram analyzer will automatically open in your default web browser.
#   
**Using the Histogram Analyzer:**

1. Event Summary (Top-left): Displays the total number of events based on current selection. Lists all individual event types: beginning with the primary signal (Z + Dark Matter), followed by the four background signals.

2. Significance Calculation: Directly below the event summary, the calculated statistical significance for the "Z + Dark Matter" signal is shown.

3. Pie Chart Visualization (Top-right): Illustrates the relative proportions of event types after applying selection cuts, offering a clear graphical representation.

4. Event Selection: All event types are included by default at startup. Use the green toggle buttons to include or exclude specific event types from the histograms.

5. Applying Cuts: Adjust the sliders to apply selection criteria (cuts) on various features.

6. Reset Filters: Each histogram has a reset button (top-right corner) to revert cuts applied to that particular feature.

8. Lepton Charge Filter (Bottom-right): Use the dedicated toggle buttons to include or exclude events based on the sum of lepton charges.

9. B-Jets Filter (Bottom-right): Use the dedicated toggle buttons to include or exclude b-jets.

10. Color Defeciency Toggle (Top-right): Use the colorblind friendly toggle to switch to colorblind friendly version of the histogram analyzer. 

