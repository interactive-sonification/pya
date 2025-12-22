from ipywidgets import widgets
from IPython.display import display
from pya import device_info, startup, Aserver
from pya.aserver import determine_backend


class AserverGUI:

    def __init__(self):
        self.index = None
        self.input_flag = False
        self.blocksize = 1024
        self.sr = 44100

        self._init_views()
        self._update_values()
        self._update_views()
        
        self.show()  # render GUI
    
    def _init_views(self):
        # audio device selector dropdown
        self.device_selector = widgets.Dropdown(
            description="AOut Device:",
            disabled=False
        )
        def on_pyagui_device_selection_change(change):
            self.index = [
                a["index"]
                for a in self.audio_devices
                if a["name"] == change["new"] and a["maxOutputChannels"] > 0
            ][0]
            self.sr = int(self.audio_devices[self.index]["defaultSampleRate"])
            self.sr_wdg.value = self.sr
        self.device_selector.observe(on_pyagui_device_selection_change, names="value")

        # input_flag selector
        self.input_flag_wdg = widgets.Checkbox(
            description="AudioIn",
            indent=False,  # Entfernt den zusätzlichen Einzug
            layout=widgets.Layout(width="70px"),
        )
        def on_input_flag_wdg_change(change):
            self.input_flag = change["new"]
        self.input_flag_wdg.observe(on_input_flag_wdg_change, names="value")

        # sr selector
        self.sr_wdg = widgets.IntText(
            description="sr:",
            layout=widgets.Layout(width="160px"),
        )
        def on_sr_wdg_value_change(change):
            self.sr = change["new"]
        self.sr_wdg.observe(on_sr_wdg_value_change, names="value")

        # blocksize selector
        self.blocksize_wdg = widgets.IntText(
            description="bs:",
            layout=widgets.Layout(width="160px"),
        )
        def on_bs_wdg_value_change(change):
            self.blocksize = change["new"]
        self.blocksize_wdg.observe(on_bs_wdg_value_change, names="value")

        # reboot button
        self.reboot_button = widgets.Button(
            description="(re)boot", layout={"width": "75px"}
        )
        def on_reboot_button_click(change):
            # Save selected and default names before reload
            prev_name = self.backend.get_device_info_by_index(self.index)["name"]
            prev_default_name = self.backend.get_default_output_device_info()["name"]

            Aserver.shutdown_default_server()
            self.backend = determine_backend()
            new_default_name = self.backend.get_default_output_device_info()["name"]
            device_names = [d["name"] for d in self.backend.get_devices()]

            # Reset device choice if previous device disconnected or default device changed
            if prev_name not in device_names or prev_default_name != new_default_name:
                self.index = None
            else:
                # Previously selected device might have changed its index after reload
                self.index = device_names.index(prev_name)
            # Launch new server
            self._update_values()
            self._update_views()
        self.reboot_button.on_click(on_reboot_button_click)

        # test tone button
        self.test_tone_button = widgets.Button(
            description="test tone", layout={"width": "70px"}
        )
        def on_test_tone_button_click(change):
            from pya.agen.lib import SinOsc, Line

            (SinOsc(800) * Line(0.1, 0, 0.2)).dup(2).play()
        self.test_tone_button.on_click(on_test_tone_button_click)

        # scope button
        self.scope_button = widgets.Button(
            description="ScopeGUI", layout={"width": "80px"}
        )
        def on_scope_button_click(change):
            Aserver.default.scope_gui()
        self.scope_button.on_click(on_scope_button_click)

        # stop button
        def on_pyagui_stop_button_click(b):
            Aserver.default.stop()
        self.stop_button = widgets.Button(
            description="Stop",
            tooltip="Stop all scheduled events on AServer",
            layout=widgets.Layout(width="60px"),
        )
        self.stop_button.on_click(on_pyagui_stop_button_click)

        self.all_widgets = widgets.HBox(
            [
                self.device_selector,
                self.sr_wdg,
                self.blocksize_wdg,
                self.input_flag_wdg,
                self.reboot_button,
                self.test_tone_button,
                self.scope_button,
                self.stop_button,
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                align_items="stretch",
                width="100%",
            ),
        )

    def _update_views(self):
        # audio device selector dropdown
        self.device_selector.options = [
            d["name"] for d in self.audio_devices if d["maxOutputChannels"] > 0
        ]
        self.device_selector.value = self.audio_devices[self.index]["name"]
        # Input flag checkbox
        self.input_flag_wdg.value = self.input_flag
        # Sample rate input
        self.sr_wdg.value = self.sr
        # Block size input
        self.blocksize_wdg.value = self.blocksize
    def _update_values(self):
        self.backend = startup(
                sr=int(self.sr),
                device=self.index,
                input_flag=self.input_flag,
                bs=self.blocksize
        ).backend
        self.audio_devices = self.backend.get_devices()
        if self.index is None:
            self.index = self.backend.get_default_output_device_info()["index"]

    def show(self):
        display(self.all_widgets)


class AGenPlayGUI:

    def __init__(self):

        # stop button
        def on_pyagui_stop_button_click(b):
            # TODO: change to Aserver.default
            if "s" in globals():
                s.stop()
            else:
                print("no pya server.")

        self.stop_button = widgets.Button(
            description="Stop",
            tooltip="Stop all scheduled events on AServer",
            layout=widgets.Layout(width="100px"),
        )
        self.stop_button.on_click(on_pyagui_stop_button_click)

        # scope selector drowpown
        def on_pyagui_scope_selection_change(change):
            # TODO: change to Aserver.default
            if "s" in globals():
                s.scope.set_mode(change["new"])
            else:
                print("no pya server")

        self.mode_selector = widgets.Dropdown(
            options=["signal", "spectrum"],
            value="signal",
            description="Mode:",
            disabled=False,
        )
        self.mode_selector.observe(on_pyagui_scope_selection_change, names="value")

        self.pyagui_gui_hbox = widgets.HBox([self.mode_selector, self.stop_button])

        def _play_with_jupyter_gui(agen, *args, **kwargs):
            agen.play(*args, **kwargs)
            display(self.pyagui_gui_hbox)

        from pya.agen.core import AGen

        AGen.playx = _play_with_jupyter_gui
