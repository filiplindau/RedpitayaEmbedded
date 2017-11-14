# -*- coding:utf-8 -*-
"""
Created on Nov 08, 2017

@author: Filip Lindau
"""
import os
import mmap
import iio
import numpy as np
from ctypes import c_uint32, c_int32, Structure, Array
import threading
import time
from copy import copy

import logging

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.CRITICAL)


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
    osc_base = 0x40100000
    buffer_size = 2 ** 14
    _DWr = (1 << (16-1)) - 1   # Fixed point range (32767)
    _CWr = 2 ** 31              # Counter range

    dec_dict = {0: 1,
                1: 8,
                2: 64,
                3: 1024,
                4: 8192,
                5: 16384}

    osc_samplerate = 125e6

    class _regset_t(Structure):
        _fields_ = [("config", c_uint32),
                    ("trg_src", c_uint32),
                    ("thr_cha", c_uint32),
                    ("thr_chb", c_uint32),
                    ("trg_dly", c_uint32),
                    ("dec", c_uint32),
                    ("w_ptr", c_uint32),
                    ("trg_ptr", c_uint32),
                    ("hys_cha", c_uint32),
                    ("hys_chb", c_uint32),
                    ("other", c_uint32),
                    ("pre_trg", c_uint32)]

    class _buffer_t(Array):
        _length_ = 2 ** 14
        _type_ = c_int32

    def __init__(self, input_range=1.0, bit_file="/opt/redpitaya/fpga/v0.94/fpga.bit"):
        self.mem_path = "/dev/mem"
        self.overlay_name = bit_file

        self.redpitaya_data = RedPitayaData()
        self.input_range = input_range

        self.lock = threading.Lock()
        self.dt_size = 10
        self.dt_array = np.zeros(self.dt_size)
        self.dt_index = 0
        self.t0 = time.time()

        self.trigger_thread = None
        self.stop_trigger_thread_flag = False
        self.trigger_time = 0.0
        self.trigger_timeout = 0.2
        self.trigger_src_w = 2
        self.reset_pending = False

        self.mem_file = None
        self.ctl_mmap = None
        self.osc0_mmap = None
        self.osc1_mmap = None
        self.ams_mmap = None
        self.osc0_buffer = None
        self.osc1_buffer = None
        self.regset = None
        self.iio_context = None
        self.iio_device = None
        self.iio_temp_channel = None
        self.iio_temp_offset = None
        self.iio_temp_scale = None
        self.adc_scale = self.input_range / np.float(2 ** (14 - 1))
        self.connected = False
        self.overlay_loaded_flag = False
        self.connect()
        self.init_scope()

    def load_overlay(self):
        """
        Load FPGA bit file by writing to /dev/xdevcfg
        :return:
        """
        root.debug("Loading overlay {0} into FPGA".format(self.overlay_name))
        os.system('cat {0} > /dev/xdevcfg'.format(self.overlay_name))
        self.overlay_loaded_flag = True

    def connect(self):
        root.debug("Entering connect")
        if self.overlay_loaded_flag is False:
            self.load_overlay()

        if self.connected is True:
            root.debug("Was already connected, close first")
            self.close()

        root.debug("Opening file to memory")
        try:
            self.mem_file = open(self.mem_path, 'r+b')
        except OSError as e:
            raise IOError(e.errno, "Opening {}: {}".format(self.mem_path, e.strerror))

        root.debug("Creating memory maps")
        self.ctl_mmap = mmap.mmap(fileno=self.mem_file.fileno(),
                                  length=0x100,
                                  offset=self.osc_base,
                                  flags=mmap.MAP_SHARED,
                                  prot=mmap.PROT_READ | mmap.PROT_WRITE)

        self.osc0_mmap = mmap.mmap(fileno=self.mem_file.fileno(),
                                   length=0x10000,
                                   offset=self.osc_base + 0x10000,
                                   flags=mmap.MAP_SHARED,
                                   prot=mmap.PROT_READ | mmap.PROT_WRITE)

        self.osc1_mmap = mmap.mmap(fileno=self.mem_file.fileno(),
                                   length=0x10000,
                                   offset=self.osc_base + 0x20000,
                                   flags=mmap.MAP_SHARED,
                                   prot=mmap.PROT_READ | mmap.PROT_WRITE)

        self.regset = self._regset_t.from_buffer(self.ctl_mmap)
        self.osc0_buffer = self._buffer_t.from_buffer(self.osc0_mmap)
        self.osc1_buffer = self._buffer_t.from_buffer(self.osc1_mmap)

        self.iio_context = iio.Context()
        self.iio_device = self.iio_context.devices[2]
        self.iio_temp_channel = self.iio_device.channels[0]
        self.iio_temp_offset = int(self.iio_temp_channel.attrs['offset'].value)
        self.iio_temp_scale = float(self.iio_temp_channel.attrs['scale'].value) / 1000.0

        self.connected = True

    def close(self):
        root.debug("Entering close")
        self.stop()
        if self.trigger_thread is not None:
            self.trigger_thread.join(0.1)
        try:
            self.ctl_mmap.close()
        except BufferError:
            root.debug("Buffer error when closing ctl_mmap")
        try:
            self.osc0_mmap.close()
        except BufferError:
            root.debug("Buffer error when closing osc0_mmap")
        try:
            self.osc1_mmap.close()
        except BufferError:
            root.debug("Buffer error when closing osc1_mmap")
        try:
            self.mem_file.close()
        except BufferError:
            root.debug("Buffer error when closing mem_file")
        except IOError:
            root.debug("IOerror when closing mem_file")
        self.connected = False

    def __del__(self):
        self.close()

    def start(self):
        """
        Start trigger_wait thread
        :return: True if successful
        """
        root.debug("In start: Starting scope")
        if self.connected is False:
            return False
        if self.trigger_thread is not None:
            self.stop()
            self.trigger_thread.join(0.1)
            root.debug("In start: Thread stopped")
        if self.redpitaya_data.triggerMode != "DISABLE":
            with self.lock:
                self.regset.trg_src = self.trigger_src_w
                self.regset.config = 1
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
        """
        Reset FPGA state machine
        :return: True if successful
        """
        if self.connected is False:
            return False
        with self.lock:
            self.regset.config = 1
        return True

    def stop(self):
        """
        Stop trigger wait thread
        :return: True if successful
        """
        if self.connected is False:
            return False
        self.reset()
        with self.lock:
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
            elif sp in ['ch2', 'channel2', '2']:
                sp = "channel2"
            elif sp in ['ext', 'external']:
                sp = 'external'
            else:
                raise ValueError(''.join(('Wrong trigger source ', str(source), ', use ch1, ch2, or ext')))
            self.redpitaya_data.triggerSource = sp
            edge = self.redpitaya_data.triggerEdge
        self.set_triggeredge(edge)

    def get_triggersource(self):
        with self.lock:
            x = copy(self.redpitaya_data.triggerSource)
        return x

    def set_triggeredge(self, edge):
        root.debug("Entering set_triggeredge")
        ed = str(edge).lower()
        with self.lock:
            sp = self.redpitaya_data.triggerSource
            if ed in ['rising', 'rise', 'r', '0', 'pe']:
                ed = "pe"
                if sp == "channel1":
                    self.trigger_src_w = 2
                elif sp == "channel2":
                    self.trigger_src_w = 4
                else:
                    self.trigger_src_w = 6
            elif ed in ['falling', 'fall', 'f', '1', 'ne']:
                ed = "ne"
                if sp == "channel1":
                    self.trigger_src_w = 3
                elif sp == "channel2":
                    self.trigger_src_w = 5
                else:
                    self.trigger_src_w = 7
            else:
                raise ValueError(''.join(('Wrong trigger edge ', str(edge), ', use rising, or falling')))
            self.reset_pending = True
            self.redpitaya_data.triggerEdge = ed

    def get_triggeredge(self):
        if self.redpitaya_data.triggerEdge == 'pe':
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

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
        with self.lock:
            x = copy(self.redpitaya_data.triggerMode)
        return x

    def set_record_length(self, rec_length):
        root.debug("Entering set_record_length")
        if rec_length > 16384:
            rec_length = 16384
        elif rec_length < 1:
            rec_length = 1
        with self.lock:
            self.redpitaya_data.recordLength = rec_length
            self.reset_pending = True
        self.set_triggerdelay_time(self.redpitaya_data.triggerDelayTime)

    def get_record_length(self):
        with self.lock:
            x = copy(self.redpitaya_data.recordLength)
        return x

    def set_decimation_factor(self, dec_index):
        root.debug("Entering set_decimation_factor")
        # Base sampling rate 125 MSPS. It is decimated by decimationfactor according to:
        if dec_index > 5:
            dec_index = 5
        elif dec_index < 0:
            dec_index = 0

        with self.lock:
            root.debug("Setting index {0}, factor {1}".format(dec_index, self.dec_dict[dec_index]))
            self.regset.dec = self.dec_dict[dec_index]
            self.redpitaya_data.decimationFactor = self.dec_dict[dec_index]
            self.redpitaya_data.sampleRate = self.osc_samplerate / self.dec_dict[dec_index]
            root.debug("Sample rate {0}".format(self.redpitaya_data.sampleRate))
        self.set_triggerdelay_time(self.redpitaya_data.triggerDelayTime)
        self.reset_pending = True

    def get_decimation_factor(self):
        with self.lock:
            x = copy(self.redpitaya_data.decimationFactor)
        return x

    def generate_timevector(self):
        root.debug("Entering generate_timevector")
        dt = 1 / self.redpitaya_data.sampleRate
        min_t = self.redpitaya_data.triggerDelayTime - self.redpitaya_data.recordLength * dt / 2
        max_t = self.redpitaya_data.triggerDelayTime + self.redpitaya_data.recordLength * dt / 2
        with self.lock:
            self.redpitaya_data.timevector = np.linspace(min_t, max_t, self.redpitaya_data.recordLength)

    def get_timevector(self):
        with self.lock:
            x = copy(self.redpitaya_data.timevector)
        return x

    def get_samplerate(self):
        with self.lock:
            x = copy(self.redpitaya_data.sampleRate)
        return x

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
            self.regset.thr_cha = np.uint16(trig_low / self.adc_scale)
            self.regset.thr_chb = np.uint16(trig_low / self.adc_scale)
            self.redpitaya_data.triggerLevel = trig_low
            self.reset_pending = True

    def get_trigger_level(self):
        with self.lock:
            x = copy(self.redpitaya_data.triggerLevel)
        return x

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
            self.reset_pending = True
            self.redpitaya_data.triggerDelaySamples = trig_delay
            self.regset.trg_dly = np.uint32(trig_delay)

    def set_trigger_pos(self, trig_pos):
        with self.lock:
            self.redpitaya_data.triggerDelay = trig_pos / self.redpitaya_data.sampleRate
            self.reset_pending = True

    def get_triggerdelay_time(self):
        with self.lock:
            x = copy(self.redpitaya_data.triggerDelayTime)
        return x

    def get_triggerdelay_samples(self):
        with self.lock:
            x = copy(self.redpitaya_data.triggerDelaySamples)
        return x

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

    def get_trigger_rate(self):
        with self.lock:
            x = copy(self.redpitaya_data.fps)
        return x

    def get_fpga_temp(self):
        """
        Get the FPGA temperature from the AXI device. Read out using IIO.
        :return: Temperature in deg C
        """
        raw = int(self.iio_temp_channel.attrs['raw'].value)
        temperature = (raw + self.iio_temp_offset) * self.iio_temp_scale
        return temperature

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
            ttime = copy(self.trigger_time)
        return ttime

    def get_trigger_wait_status(self):
        with self.lock:
            trig_status = copy(self.redpitaya_data.triggerWait)
        return trig_status

    def get_osc_is_triggered(self):
        with self.lock:
            # if self.regset.config == 0 and self.regset.trg_src == 0:
            if self.regset.trg_src == 0:
                is_triggered = True
            else:
                is_triggered = False
        return is_triggered

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
        trigger_time_vec = [time.time()]
        target_fps_ave = 20
        update_data = False
        while stop is False:
            if self.reset_pending is True:
                self.reset()
                with self.lock:
                    self.reset_pending = False
            trig_status = self.get_osc_is_triggered()
            with self.lock:
                self.trigger_time = time.time() - start_time
                if trig_status is True:
                    update_data = True
                elif self.trigger_time > self.trigger_timeout:
                    self.redpitaya_data.triggerWait = True
                    if self.redpitaya_data.triggerMode == "AUTO":
                        update_data = True
                elif self.redpitaya_data.triggerMode == "FORCE":
                    self.regset.config = 1
                    update_data = True
                    stop = True
                if self.stop_trigger_thread_flag is True:
                    stop = True
                elif update_data is True:
                    # Calculate FPS:
                    start_time = time.time()
                    if len(trigger_time_vec) > target_fps_ave:
                        trigger_time_vec = trigger_time_vec[1:]
                    trigger_time_vec.append(start_time)
                    self.redpitaya_data.fps = len(trigger_time_vec) / (trigger_time_vec[-1] - trigger_time_vec[0])
                    self.trigger_time = 0.0
            if update_data is True:
                self.redpitaya_data.triggerWait = False
                t2 = time.time()
                self.update_waveforms()
                t3 = time.time()
                root.debug("Update waveforms time: {}".format(t3 - t2))
                if self.redpitaya_data.triggerMode in ["NORMAL", "AUTO"]:
                    self.regset.config = 1

                    # start_time = time.time()
                    # if len(trigger_time_vec) > target_fps_ave:
                    #     red_len = max(0, len(trigger_time_vec) - target_fps_ave + 1)
                    #     trigger_time_vec = trigger_time_vec[red_len:]
                    # trigger_time_vec.append(start_time)
                    # with self.lock:
                    #     try:
                    #         self.redpitaya_data.fps = len(trigger_time_vec) / (
                    #         trigger_time_vec[-1] - trigger_time_vec[0])
                    #     except ZeroDivisionError:
                    #         self.redpitaya_data.fps = 0.0
                    #     est_fps_ave = self.redpitaya_data.fps * self.trigger_timeout * 2
                    #     if not (0.9 < abs(est_fps_ave - target_fps_ave) / target_fps_ave < 1.1):
                    #         target_fps_ave = int(est_fps_ave)
                    #         root.debug("New target fps averaging length: {0}".format(target_fps_ave))
                    # self.trigger_time = 0.0

                    # Wait this time to make sure the pre trig buffer is filled before a trig occurs:
                    wait_time = (self.redpitaya_data.recordLength -
                                 self.redpitaya_data.triggerDelaySamples) / self.redpitaya_data.sampleRate
                    root.debug("Sleeping {0} ms".format(1000*(wait_time - (time.time() - start_time))))
                    time.sleep(max(0.0, wait_time - (time.time() - start_time)))
                    self.regset.trg_src = self.trigger_src_w
                elif self.redpitaya_data.triggerMode in ["SINGLE"]:
                    stop = True
                update_data = False
            time.sleep(0.001)
        root.debug("Exiting wait_for_trigger")
        self.trigger_time = 0.0
        self.stop()
        with self.lock:
            self.redpitaya_data.fps = 0.0

    def update_waveforms(self):
        # root.debug("Entering update_waveforms")
        with self.lock:
            sig1 = self.data(0, self.redpitaya_data.recordLength)
            sig2 = self.data(1, self.redpitaya_data.recordLength)
            self.redpitaya_data.waveform1 = sig1
            self.redpitaya_data.waveform2 = sig2
            self.redpitaya_data.triggerCounter1 += 1
            self.redpitaya_data.triggerCounter2 += 1
            # t = time.time()
            # dt = t - self.t0
            # self.dt_array[self.dt_index % self.dt_size] = dt
            # if self.dt_index > self.dt_size:
            #     self.redpitaya_data.fps = 1 / self.dt_array.mean()
            # else:
            #     self.redpitaya_data.fps = 1 / self.dt_array[0:self.dt_index]
            # self.t0 = t
            # self.dt_index += 1

    def data(self, osc_id=0, siz: int = buffer_size):
        trig_delay = np.int32(self.redpitaya_data.triggerDelaySamples)
        # ptr = self.regset.trg_ptr + trig_delay
        ptr = (self.regset.trg_ptr + trig_delay) % self.buffer_size

        t0 = time.time()
        # if osc_id == 0:
        #     wave = np.roll(self.osc0_buffer, -ptr)
        # else:
        #     wave = np.roll(self.osc1_buffer, -ptr)
        if osc_id == 0:
            buf = np.frombuffer(self.osc0_buffer, np.int32)
        else:
            buf = np.frombuffer(self.osc1_buffer, np.int32)
        wave = np.zeros(siz)

        # ptr always > 0 after mod
        if siz > ptr:
            b_i0 = self.buffer_size - (siz - ptr)
            b_i1 = self.buffer_size
            w_i0 = 0
            w_i1 = b_i1 - b_i0

            b_i2 = 0
            b_i3 = ptr
            w_i2 = w_i1
            w_i3 = siz
        else:
            b_i0 = ptr - siz
            b_i1 = ptr
            w_i0 = 0
            w_i1 = b_i1 - b_i0

            b_i2 = 0
            b_i3 = 0
            w_i2 = 0
            w_i3 = 0

        root.debug("w_i0 {0}, w_i1 {1}, b_i0 {2}, b_i1 {3}".format(w_i0, w_i1, b_i0, b_i1))
        root.debug("w_i2 {0}, w_i3 {1}, b_i2 {2}, b_i3 {3}".format(w_i2, w_i3, b_i2, b_i3))
        wave[w_i0:w_i1] = buf[b_i0:b_i1]
        wave[w_i2:w_i3] = buf[b_i2:b_i3]

        t1 = time.time()
        data = wave.astype("float32")[-siz:] * self.adc_scale
        t2 = time.time()
        root.debug("In data: roll time {0} ms, float time {1} ms".format(1000*(t1-t0), 1000*(t2-t1)))
        # pre_acq = self.redpitaya_data.recordLength - trig_delay - self.regset.pre_trg
        # if pre_acq > 0:
        #     data[:pre_acq] = 0
        return data


if __name__ == "__main__":
    ot = RedpitayaControl()
