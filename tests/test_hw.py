"""The hardware layer's arithmetic, with no arm attached."""

import numpy as np
import pytest

from hw import feetech as ft
from hw.mirror import lag_steps, speed_limit
from hw.so101 import RAD_PER_TICK, SO101


class FakeSerial:
    """Records what is written and replays a canned reply."""

    def __init__(self, reply: bytes = b""):
        self.sent, self.reply = b"", reply

    def reset_input_buffer(self):
        pass

    def write(self, b):
        self.sent += b

    def read(self, n):
        out, self.reply = self.reply[:n], self.reply[n:]
        return out


def bus(reply=b""):
    b = ft.Bus.__new__(ft.Bus)
    b.ser = FakeSerial(reply)
    return b


def status(sid, params=b"", err=0):
    body = bytes([sid, len(params) + 2, err]) + params
    return b"\xff\xff" + body + bytes([~sum(body) & 0xFF])


def test_read_packet_and_checksum():
    b = bus(status(3, (2048).to_bytes(2, "little")))
    assert b.read(3, ft.POSITION) == 2048
    # FF FF id=3 len=4 READ addr=56 size=2 checksum
    assert b.ser.sent == bytes([0xFF, 0xFF, 3, 4, 0x02, 56, 2, ~(3 + 4 + 2 + 56 + 2) & 0xFF])


def test_bad_checksum_is_refused():
    reply = bytearray(status(1, b"\x00\x08"))
    reply[-1] ^= 0xFF
    with pytest.raises(ft.BusError):
        bus(bytes(reply)).read(1, ft.POSITION)


def test_sync_write_is_one_broadcast_packet():
    b = bus()
    b.sync_write(ft.GOAL_POSITION, {1: 100, 2: 4000})
    assert b.ser.sent[:5] == bytes([0xFF, 0xFF, 0xFE, 2 + 2 + 2 * 3, 0x83])
    assert b.ser.sent[5:7] == bytes([42, 2])


@pytest.mark.parametrize("v", [0, 1, 2047, -1, -2047, -895])
def test_sign_magnitude_round_trip(v):
    assert ft.decode_sm(ft.encode_sm(v, 11), 11) == v


def test_load_decodes_sign_in_bit_10():
    assert ft.decode_sm(1136, 10) == -112
    assert ft.decode_sm(112, 10) == 112


def test_ticks_and_radians_are_inverse():
    arm = SO101.__new__(SO101)
    arm.center = np.full(5, 2048.0)
    arm.sign = np.array([1, -1, 1, 1, -1], float)
    arm.lo, arm.hi = np.zeros(5), np.full(5, 4095.0)
    q = np.array([0.3, -1.2, 1.5, -0.4, 2.0])
    assert np.allclose(arm.to_rad(arm.to_ticks(q)), q, atol=RAD_PER_TICK)
    assert arm.to_ticks(np.zeros(5)).tolist() == [2048] * 5


def test_ticks_clip_to_the_limits():
    arm = SO101.__new__(SO101)
    arm.center, arm.sign = np.full(5, 2048.0), np.ones(5)
    arm.lo, arm.hi = np.full(5, 1000.0), np.full(5, 3000.0)
    assert arm.to_ticks(np.full(5, 3.0)).tolist() == [3000] * 5


def test_speed_limit_caps_steps_and_reports_it():
    target = np.zeros((4, 5))
    target[2:, 0] = 1.0  # a 1 rad jump
    out, bit = speed_limit(target, dt=0.1, vmax=2.0)
    assert np.allclose(np.diff(out[:, 0]), [0, 0.2, 0.2])
    assert bit == pytest.approx(2 / 3)


def test_lag_is_found():
    t = np.arange(300) * 0.033
    cmd = np.sin(2 * t)
    meas = np.r_[np.full(4, cmd[0]), cmd[:-4]]
    assert lag_steps(cmd, meas) == 4
