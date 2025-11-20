from rtmidi import MidiIn

midi_msg_types = {
    # Channel Voice Messages
    0x80: "NoteOff",
    0x90: "NoteOn",
    0xA0: "PolyphonicKeyPressure",  # Key Aftertouch
    0xB0: "ControlChange",
    0xC0: "ProgramChange",
    0xD0: "ChannelPressure",  # Channel Aftertouch
    0xE0: "PitchBendChange",
    # System Common Messages
    0xF0: "SystemExclusiveStart",
    0xF1: "MTCQuarterFrame",
    0xF2: "SongPositionPointer",
    0xF3: "SongSelect",
    0xF6: "TuneRequest",
    0xF7: "SystemExclusiveEnd",
    # System Real-Time Messages
    0xF8: "TimingClock",
    0xFA: "Start",
    0xFB: "Continue",
    0xFC: "Stop",
    0xFE: "ActiveSensing",
    0xFF: "SystemReset",
}


def cancel_MidiIn_callbacks_and_close_ports(verbose=True):
    "cancel callbacks and close ports for MidiIn"
    import gc

    for i, o in enumerate(gc.get_objects()):
        if isinstance(o, MidiIn):
            if verbose:
                print(i, "close:", o)
            o.cancel_callback()
            o.close_port()


def parse_status_byte(status):
    "parse to (msg_type (str), channel (int or None)."
    # data byte
    if status < 0x80:
        return "DataByte", None

    # channel voice
    if 0x80 <= status <= 0xEF:  # Channel voice
        msg_type = status & 0xF0
        channel = status & 0x0F
        return midi_msg_types.get(msg_type, "Unknown"), channel

    # System messages (F0–FF)
    return midi_msg_types.get(status, "UnknownSystemMessage"), None


class MIDI_ctrl:

    def __init__(self, midiidx=0, verbose=True):
        self.midiin = MidiIn()
        midi_ports = self.midiin.get_ports()  # get list of available MIDI ports
        print("Ports:", midi_ports)
        self.midiin.open_port(midiidx)
        self.octave = 1
        self.verbose = verbose
        self.level = 0

        self.callbacks = dict(
            NoteOn=lambda x: None,
            NoteOff=lambda x: None,
        )

        def midi_callback(event, ctx=self):
            msg, dt = event
            verbose = ctx.verbose

            # extract MIDI data from msg
            status = msg[0]
            name, chn = parse_status_byte(status)

            if verbose:
                print(
                    f"MIDI {name:21}: CH{chn:02}: {msg[1:]} (dt={dt:10.3})",
                    end="    \r",
                )

            # process MIDI message
            if name == "NoteOn" or name == "NoteOff":
                note, vel = msg[1:3]
                try:
                    fn = ctx.callbacks[name]
                    fn(note=note, vel=vel, chn=chn, ctx=self)
                except ValueError:
                    pass

        # register callback function
        self.midiin.set_callback(midi_callback, self)
        print("registered.")

    def __del__(self):
        self.midiin.cancel_callback()
        self.midiin.close_port()

    def set_callback(self, msg_name="NoteOn", fn=lambda x: None):
        self.callbacks[msg_name] = fn

    def clear_callback(self, msg_name="NoteOn"):
        self.callbacks[msg_name] = lambda x: None
