# heatmap_canvas.py

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import logging


class HeatmapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.colorbar = None  # Initialize colorbar reference
        super(HeatmapCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # Optional: Adjust layout
        self.fig.tight_layout()

    def plot_heatmap(self, data, mask=None, cmap='jet'):
        """
        Plot the heatmap using matplotlib's imshow with a dedicated colorbar axes.

        Args:
            data (np.ndarray): 2D array representing the heatmap.
            mask (np.ndarray, optional): Boolean array for masking. Defaults to None.
            cmap (str, optional): Colormap for the heatmap. Defaults to 'jet'.
        """
        try:
            self.axes.clear()

            if mask is not None:
                # Apply mask by masking invalid data
                data = np.ma.masked_where(~mask, data)

            im = self.axes.imshow(data, cmap=cmap, interpolation='nearest', aspect='auto')
            self.axes.axis('off')  # Hide axes for a cleaner look

            # Set the color for masked areas to black
            im.cmap.set_bad(color='black')

            # Remove existing colorbar if it exists and is valid
            if self.colorbar is not None:
                try:
                    self.colorbar.remove()
                    #logging.debug("Removed existing colorbar.")
                except AttributeError:
                    #logging.warning("Colorbar already removed or was never set.")
                    pass
                except Exception as e:
                    logging.error(f"Failed to remove existing colorbar: {e}")
                finally:
                    self.colorbar = None

            # Create a new axes for the colorbar using make_axes_locatable
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            self.colorbar = self.fig.colorbar(im, cax=cax, orientation='vertical')
            #logging.debug("Added new colorbar.")

            self.draw()
        except Exception as e:
            logging.error(f"Failed to plot heatmap: {e}")
