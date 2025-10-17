from ipywidgets import widgets
from IPython.display import display
from pya import device_info, startup, Aserver
from pya.aserver import determine_backend


class AserverGUI:

    def __init__(self):
        self.audio_devices = device_info(verbose=False)
        self.blocksize = 1024
        self.sr = 44100
        self.index = determine_backend().get_default_output_device_info()['index']

        # audio device selector dropdown
        self.device_selector = widgets.Dropdown(
            options=[
                d["name"] for d in self.audio_devices if d["maxOutputChannels"] > 0
            ],
            description="AOut Device:",
            disabled=False,
            value=self.audio_devices[self.index]['name'],
        )

        def on_pyagui_device_selection_change(change):
            self.index = [
                a["index"]
                for a in self.audio_devices
                if a["name"] == change["new"] and a["maxOutputChannels"] > 0
            ][0]
            self.sr = int(self.audio_devices[self.index]["defaultSampleRate"])
            self.sr_wdg.set_state({"value": self.sr})

        self.device_selector.observe(on_pyagui_device_selection_change, names="value")

        # sr  selector
        self.sr_wdg = widgets.IntText(
            description="sr:",
            value=self.sr,
            layout=widgets.Layout(width="160px"),
        )

        def on_sr_wdg_value_change(change):
            self.sr = change["new"]

        self.sr_wdg.observe(on_sr_wdg_value_change, names="value")

        # blocksize selector
        self.blocksize_wdg = widgets.IntText(
            description="bs:",
            value=self.blocksize,
            layout=widgets.Layout(width="160px"),
        )

        def on_bs_wdg_value_change(change):
            self.blocksize = change["new"]

        self.blocksize_wdg.observe(on_bs_wdg_value_change, names="value")

        # reboot button
        self.reboot_button = widgets.Button(
            description="reboot", layout={"width": "60px"}
        )

        def on_reboot_button_click(change):
            # TODO: change to Aserver.default
            global s
            if "s" in globals() and isinstance(s, Aserver):
                s.shutdown_default_server()
            s = startup(sr=int(self.sr), device=self.index, bs=self.blocksize)

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
            # TODO: change to Aserver.default
            global s
            if "s" in globals() and isinstance(s, Aserver):
                s.scope_gui()

        self.scope_button.on_click(on_scope_button_click)

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
            layout=widgets.Layout(width="60px"),
        )
        self.stop_button.on_click(on_pyagui_stop_button_click)

        # display GUI
        display(
            widgets.HBox(
                [
                    self.device_selector,
                    self.sr_wdg,
                    self.blocksize_wdg,
                    self.reboot_button,
                    self.test_tone_button,
                    self.scope_button,
                    self.stop_button,
                ]
            )
        )

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
            description='Stop',
            tooltip='Stop all scheduled events on AServer',
            layout=widgets.Layout(width='100px')
        )
        self.stop_button.on_click(on_pyagui_stop_button_click)

        # scope selector drowpown
        def on_pyagui_scope_selection_change(change):
            # TODO: change to Aserver.default
            if 's' in globals():
                s.scope.set_mode(change['new'])
            else:
                print("no pya server")
        self.mode_selector = widgets.Dropdown(
            options=['signal', 'spectrum'],
            value='signal',
            description='Mode:',
            disabled=False,
        )
        self.mode_selector.observe(on_pyagui_scope_selection_change, names='value')

        self.pyagui_gui_hbox = widgets.HBox([self.mode_selector, self.stop_button])

        def _play_with_jupyter_gui(agen, *args, **kwargs):
            agen.play(*args, **kwargs)
            display(self.pyagui_gui_hbox)

        from pya.agen.core import AGen
        AGen.playx = _play_with_jupyter_gui