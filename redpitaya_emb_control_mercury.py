# -*- coding:utf-8 -*-
"""
Created on Oct 31, 2017

@author: Filip Lindau
"""
import numpy as np
import threading
import time
from redpitaya.overlay.mercury import mercury as overlay

import logging

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.DEBUG)


class RedPitayaData(object):
    def __init__(self):
        self.triggerMode = 'NORMAL'
        self.triggerSource = 'CHANNEL1'
        self.triggerEdge = 'pe'
        self.triggerLevel = 0.0
        self.triggerDelayTime = 0.0
        self.triggerDelaySamples = 4096
        self.triggerWait = False
        self.recordLength = 8192
        self.decimationFactor = 1
        self.sampleRate = 125e6
        self.waveform1 = np.zeros(2000)
        self.triggerCounter1 = 0
        self.waveform2 = np.zeros(2000)
        self.triggerCounter2 = 0
        self.waveformDatatype = np.float32
        self.timevector = np.zeros(2000)
        self.fps = 0.0


class RedpitayaControl(object):
    def __init__(self):
        self.redpitaya_data = RedPitayaData()

        self.lock = threading.Lock()
        self.dt_size = 10
        self.dt_array = np.zeros(self.dt_size)
        self.dt_index = 0
        self.t0 = time.time()

        self.fgpa = None
        self.osc0 = None
        self.osc1 = None
        self.osc = None
        self.connected = False
        self.connect()

        self.trigger_thread = None
        self.stop_trigger_thread_flag = False
        self.trigger_time = 0.0
        self.trigger_timeout = 0.2
        self.reset_pending = False

    def connect(self):
        if self.connected is False:
            self.fpga = overlay()
            self.connected = True
            try:
                self.osc0 = self.fpga.osc(0, 20.0)
                self.osc1 = self.fpga.osc(1, 20.0)
                self.osc = [self.osc0, self.osc1]
            except BlockingIOError:
                self.connected = False

    def close(self):
        pass

    def start(self):
        root.debug("In start: Starting scope")
        if self.connected is False:
            return False
        if self.trigger_thread is not None:
            self.stop()
            self.trigger_thread.join(0.1)
            root.debug("In start: Thread stopped")
        if self.redpitaya_data.triggerMode != "DISABLE":
            with self.lock:
                for osc in self.osc:
                    osc.start()
                self.redpitaya_data.triggerWait = False
                self.stop_trigger_thread_flag = False
            self.trigger_thread = threading.Thread()
            threading.Thread.__init__(self.trigger_thread, target=self.wait_for_trigger)
            self.trigger_thread.start()
            root.debug("In start: Thread started")
            return True
        else:
            return False

    def reset(self):
        if self.connected is False:
            return False
        with self.lock:
            for osc in self.osc:
                osc.reset()
        return True

    def stop(self):
        if self.connected is False:
            return False
        with self.lock:
            for osc in self.osc:
                osc.stop()
            if self.trigger_thread is not None:
                self.stop_trigger_thread_flag = True
        return True

    def init_scope(self):
        self.set_trigger_source(self.redpitaya_data.triggerSource)
        self.set_triggermode(self.redpitaya_data.triggerMode)
        self.set_trigger_level(self.redpitaya_data.triggerLevel)
        self.set_record_length(self.redpitaya_data.recordLength)
        self.set_decimation_factor(self.redpitaya_data.decimationFactor)

    def set_trigger_source(self, source):
        """ Set the trigger source to use.
        Input: ch1, ch2 or ext
        """
        root.debug("Entering set_trigger_source")
        sp = str(source).lower()
        with self.lock:
            if sp in ['ch1', 'channel1', '1']:
                sp = "channel1"
                for osc in self.osc:
                    osc.trig_src = self.fpga.trig_src["osc0"]
                    osc.sync_src = self.fpga.sync_src["osc0"]
            elif sp in ['ch2', 'channel2', '2']:
                sp = "channel2"
                for osc in self.osc:
                    osc.trig_src = self.fpga.trig_src["osc1"]
                    osc.sync_src = self.fpga.sync_src["osc0"]
            elif sp in ['ext', 'external']:
                sp = 'external'
                for osc in self.osc:
                    osc.trig_src = self.fpga.trig_src["la"]
                    osc.sync_src = self.fpga.sync_src["osc0"]
            else:
                raise ValueError(''.join(('Wrong trigger source ', str(source), ', use ch1, ch2, or ext')))
            self.redpitaya_data.triggerSource = sp
        edge = self.redpitaya_data.triggerEdge
        self.set_triggeredge(edge)

    def get_triggersource(self):
        return self.redpitaya_data.triggerSource

    def set_triggermode(self, mode):
        root.debug("Entering set_triggermode")
        sp = str(mode).lower()
        if sp in ['normal', 'norm', 'n']:
            sp = 'NORMAL'
        elif sp in ['single', 'sing', 's']:
            sp = 'SINGLE'
        elif sp in ['force', 'f']:
            sp = 'FORCE'
        elif sp in ['auto', 'a']:
            sp = 'AUTO'
        elif sp in ['disable', 'd']:
            sp = 'DISABLE'
        else:
            raise ValueError(
                ''.join(('Wrong trigger mode ', str(mode), ', use normal, single, force, auto, or disable')))
        with self.lock:
            self.redpitaya_data.triggerMode = sp
        if sp == "DISABLE":
            self.stop()

    def get_triggermode(self):
        return self.redpitaya_data.triggerMode

    def set_triggeredge(self, edge):
        root.debug("Entering set_triggeredge")
        sp = str(edge).lower()
        with self.lock:
            if sp in ['rising', 'rise', 'r', '0', 'pe']:
                ed = "pe"
                for osc in self.osc:
                    osc.edge = "pos"
            elif sp in ['falling', 'fall', 'f', '1', 'ne']:
                ed = "ne"
                for osc in self.osc:
                    osc.edge = "neg"
            else:
                raise ValueError(''.join(('Wrong trigger edge ', str(edge), ', use rising, or falling')))
            self.reset_pending = True
            self.redpitaya_data.triggerEdge = ed

    def get_triggeredge(self):
        if self.redpitaya_data.triggerEdge == 'pe':
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

    def set_record_length(self, rec_length):
        root.debug("Entering set_record_length")
        if rec_length > 16384:
            rec_length = 16384
        elif rec_length < 1:
            rec_length = 1
        # with self.lock:
        #     for osc in self.osc:
        #         osc.buffer_size = rec_length
        with self.lock:
            self.redpitaya_data.recordLength = rec_length
            self.reset_pending = True
        self.set_triggerdelay_time(self.redpitaya_data.triggerDelayTime)

    def get_record_length(self):
        return self.redpitaya_data.recordLength

    def set_decimation_factor(self, dec_factor):
        root.debug("Entering set_decimation_factor")
        # Base sampling rate 125 MSPS. It is decimated by decimationfactor according to:
        decDict = {0: 1,
                   1: 8,
                   2: 64,
                   3: 1024,
                   4: 8192,
                   5: 16384}
        if dec_factor > self.redpitaya_data.recordLength:
            dec_factor = self.redpitaya_data.recordLength
        elif dec_factor < 1:
            dec_factor = 1

        with self.lock:
            for osc in self.osc:
                osc.decimation = dec_factor
            self.redpitaya_data.decimationFactor = dec_factor
            self.redpitaya_data.sampleRate = self.osc[0].sample_rate
        self.set_triggerdelay_time(self.redpitaya_data.triggerDelayTime)
        self.reset_pending = True

    def get_decimation_factor(self):
        return self.redpitaya_data.decimationFactor

    def generate_timevector(self):
        root.debug("Entering generate_timevector")
        dt = 1 / self.redpitaya_data.sampleRate
        min_t = self.redpitaya_data.triggerDelayTime - self.redpitaya_data.recordLength * dt / 2
        max_t = self.redpitaya_data.triggerDelayTime + self.redpitaya_data.recordLength * dt / 2
        with self.lock:
            self.redpitaya_data.timevector = np.linspace(min_t, max_t, self.redpitaya_data.recordLength)

    def get_timevector(self):
        return self.redpitaya_data.timevector

    def get_samplerate(self):
        return self.redpitaya_data.sampleRate

    def set_trigger_level(self, trig_level):
        root.debug("Entering set_trigger_level")
        if type(trig_level) in [list, np.ndarray]:
            trig_low = trig_level[0]
            trig_high = trig_level[1]
        else:
            if trig_level > 2:
                trig_level = 2
            elif trig_level < -2:
                trig_level = -2
            trig_low = trig_level
            trig_high = trig_level + 0.1
        with self.lock:
            for osc in self.osc:
                osc.level = [trig_low, trig_high]
            self.redpitaya_data.triggerLevel = trig_low
            self.reset_pending = True

    def get_trigger_level(self):
        return self.redpitaya_data.triggerLevel

    def set_triggerdelay_time(self, trig_delay):
        """Set trigger delay in s
        """
        with self.lock:
            self.redpitaya_data.triggerDelayTime = trig_delay
        trigger_delay_samples = self.redpitaya_data.recordLength / 2.0 + trig_delay * self.redpitaya_data.sampleRate
        if trigger_delay_samples < 0:
            self.set_triggerdelay_samples(0)
        else:
            self.set_triggerdelay_samples(trigger_delay_samples)
        self.generate_timevector()

    def set_triggerdelay_samples(self, trig_delay):
        print("New trig delay: {0} samples".format(trig_delay))
        with self.lock:
            pre_trig = np.int(trig_delay)
            post_trig = np.int(self.redpitaya_data.recordLength - trig_delay)
            for osc in self.osc:
                osc.trigger_pre = pre_trig
                osc.trigger_post = post_trig
            self.reset_pending = True
            self.redpitaya_data.triggerDelaySamples = trig_delay

    def set_trigger_pos(self, trig_pos):
        with self.lock:
            pre_trig = np.int(trig_pos)
            post_trig = np.int(self.redpitaya_data.recordLength - trig_pos)
            for osc in self.osc:
                osc.trigger_pre = pre_trig
                osc.trigger_post = post_trig
            self.redpitaya_data.triggerDelay = trig_pos / self.redpitaya_data.sampleRate
            self.reset_pending = True

    def get_triggerdelay_time(self):
        return self.redpitaya_data.triggerDelayTime

    def get_triggerdelay_samples(self):
        return self.redpitaya_data.triggerDelaySamples

    def get_trigger_counter(self, channel):
        """ triggerCounter = getTriggerCounter(channel)

        input:
            channel = 1 or 2
        output:
            triggerCounter = redpitaya internal trigger counter for the specific
                             channel from the latest updateWaveform call

        Returns trigger counter for the waveform stored in the redPitayaData data structure
        """
        if channel == 1:
            return self.redpitaya_data.triggerCounter1
        elif channel == 2:
            return self.redpitaya_data.triggerCounter2
        else:
            raise ValueError(''.join(('Wrong channel ', str(channel), ', use 1, or 2')))

    def get_fpga_temp(self):
        return 0.0

    def get_waveform(self, channel):
        """ waveform = getWaveform(channel)

        input:
            channel = 1 or 2
        output:
            waveform = numpy array containing the data from the latest updateWaveform call

        Returns a waveform stored in the redPitayaData data structure
        """
        if channel == 1:
            return self.redpitaya_data.waveform1
        elif channel == 2:
            return self.redpitaya_data.waveform2
        else:
            raise ValueError(''.join(('Wrong channel ', str(channel), ', use 1, or 2')))

    def get_trigger_time(self):
        with self.lock:
            ttime = self.trigger_time
        return ttime

    def get_trigger_wait_status(self):
        with self.lock:
            trig_status = self.redpitaya_data.triggerWait
        return trig_status

    def get_running_status(self):
        # run_status = self.osc[0].status_run()
        if self.trigger_thread is not None:
            run_status = self.trigger_thread.is_alive()
        else:
            run_status = False
        return run_status

    def wait_for_trigger(self):
        root.debug("Entering wait_for_trigger")
        stop = False
        start_time = time.time()
        self.trigger_time = 0.0
        update_data = False
        while stop is False:
            with self.lock:
                if self.reset_pending is True:
                    for osc in self.osc:
                        osc.reset()
                        osc.start()
                    self.reset_pending = False
                trig_status = self.osc[0].status_trigger()
                self.trigger_time = time.time() - start_time
                if trig_status is True:
                    update_data = True
                elif self.trigger_time > self.trigger_timeout:
                    self.redpitaya_data.triggerWait = True
                    if self.redpitaya_data.triggerMode == "AUTO":
                        for osc in self.osc:
                            osc.stop()
                        update_data = True
                elif self.redpitaya_data.triggerMode == "FORCE":
                    for osc in self.osc:
                        osc.stop()
                    update_data = True
                    stop = True
                if self.stop_trigger_thread_flag is True:
                    stop = True
            if update_data is True:
                self.redpitaya_data.triggerWait = False
                self.update_waveforms()
                if self.redpitaya_data.triggerMode in ["NORMAL", "AUTO"]:
                    for osc in self.osc:
                        osc.start()
                    start_time = time.time()
                    self.trigger_time = 0.0
                elif self.redpitaya_data.triggerMode in ["SINGLE"]:
                    stop = True
                update_data = False
            time.sleep(0.001)
        root.debug("Exiting wait_for_trigger")
        self.trigger_time = 0.0
        self.stop()

    def update_waveforms(self):
        root.debug("Entering update_waveforms")
        with self.lock:
            sig1 = self.osc[0].data(self.redpitaya_data.recordLength)
            sig2 = self.osc[1].data(self.redpitaya_data.recordLength)
            self.redpitaya_data.waveform1 = sig1
            self.redpitaya_data.waveform2 = sig2
            self.redpitaya_data.triggerCounter1 += 1
            self.redpitaya_data.triggerCounter2 += 1
            t = time.time()
            dt = t - self.t0
            self.dt_array[self.dt_index % self.dt_size] = dt
            if self.dt_index > self.dt_size:
                self.redpitaya_data.fps = 1 / self.dt_array.mean()
            else:
                self.redpitaya_data.fps = 1 / self.dt_array[0:self.dt_index]
            self.t0 = t
            self.dt_index += 1


if __name__ == "__main__":
    rpc = RedpitayaControl()
    rpc.set_trigger_level([0.5, 0.6])



