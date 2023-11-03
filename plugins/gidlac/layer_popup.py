# -*- coding: utf-8 -*-
from .add_raster_layer_ui import AddRasterLayer


class popup_dialog:
    def __init__(self):
        self.AddRasterPopup = AddRasterLayer()
        self.pluginIsActive = False
        self.signals_connection()

    def run(self):
        """Run method that loads and starts the plugin"""

        if not self.pluginIsActive:
            self.pluginIsActive = True
            self.AddRasterPopup.show()

    def signals_connection(self):
        self.AddRasterPopup.buttonBox.accepted.connect(self.returnLayer)
        self.AddRasterPopup.buttonBox.rejected.connect(self.cancel)

    def returnLayer(self):
        print('clicked accept')
        currentfile = self.AddRasterPopup.comboFile.currentLayer()
        print(currentfile)

    def cancel(self):
        print('canceled')
        self.close()


