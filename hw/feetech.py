"""A minimal Feetech STS3215 bus: ping, read, write, sync read and sync write.

The SO-101's servos speak Feetech's half-duplex TTL protocol at 1 Mbaud
through the USB bus adapter. LeRobot wraps the same protocol in a much larger
stack; the five-servo, position-only arm here needs only this much of it.

Packet: FF FF id len instr params... checksum, with len = len(params) + 2 and
checksum = ~(id + len + instr + sum(params)) & 0xFF. Multi-byte values are
little-endian.
"""

from __future__ import annotations

import time

import serial

# Control table (STS3215, firmware 3.x). Address, size in bytes.
MODEL = (3, 2)
ID = (5, 1)
MIN_LIMIT = (9, 2)
MAX_LIMIT = (11, 2)
MAX_TORQUE = (16, 2)
P_GAIN = (21, 1)
D_GAIN = (22, 1)
I_GAIN = (23, 1)
HOMING_OFFSET = (31, 2)  # sign-magnitude, sign in bit 11
MODE = (33, 1)
TORQUE_ENABLE = (40, 1)
ACCELERATION = (41, 1)
GOAL_POSITION = (42, 2)
GOAL_TIME = (44, 2)
GOAL_SPEED = (46, 2)
TORQUE_LIMIT = (48, 2)
LOCK = (55, 1)
POSITION = (56, 2)
SPEED = (58, 2)  # sign in bit 15
LOAD = (60, 2)  # sign in bit 10
VOLTAGE = (62, 1)
TEMPERATURE = (63, 1)
MOVING = (66, 1)
CURRENT = (69, 2)

PING, READ, WRITE, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x82, 0x83

TICKS = 4096  # per revolution


def encode_sm(value: int, sign_bit: int) -> int:
    """Signed int -> Feetech sign-magnitude."""
    return (abs(value) | (1 << sign_bit)) if value < 0 else value


def decode_sm(raw: int, sign_bit: int) -> int:
    mag = raw & ((1 << sign_bit) - 1)
    return -mag if raw & (1 << sign_bit) else mag


class BusError(IOError):
    pass


class Bus:
    def __init__(self, port: str, baud: int = 1_000_000, timeout: float = 0.02):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.ser.reset_input_buffer()

    def close(self) -> None:
        self.ser.close()

    # -- framing ---------------------------------------------------------------

    def _send(self, sid: int, instr: int, params: bytes = b"") -> None:
        body = bytes([sid, len(params) + 2, instr]) + params
        self.ser.reset_input_buffer()
        self.ser.write(b"\xff\xff" + body + bytes([~sum(body) & 0xFF]))

    def _status(self) -> tuple[int, int, bytes]:
        """Read one status packet: (id, error byte, params)."""
        head = self.ser.read(4)
        # Resync on a leading byte of line noise.
        while len(head) == 4 and head[:2] != b"\xff\xff":
            nxt = self.ser.read(1)
            if not nxt:
                break
            head = head[1:] + nxt
        if len(head) < 4 or head[:2] != b"\xff\xff":
            raise BusError(f"no reply (got {head.hex() or 'nothing'})")
        sid, n = head[2], head[3]
        rest = self.ser.read(n)
        if len(rest) < n:
            raise BusError(f"short reply from {sid}")
        err, params, chk = rest[0], rest[1:-1], rest[-1]
        if (~(sid + n + err + sum(params)) & 0xFF) != chk:
            raise BusError(f"bad checksum from {sid}")
        return sid, err, bytes(params)

    # -- single servo ----------------------------------------------------------

    def ping(self, sid: int) -> bool:
        self._send(sid, PING)
        try:
            return self._status()[0] == sid
        except BusError:
            return False

    def read(self, sid: int, reg: tuple[int, int]) -> int:
        addr, size = reg
        self._send(sid, READ, bytes([addr, size]))
        _, _, p = self._status()
        return int.from_bytes(p[:size], "little")

    def write(self, sid: int, reg: tuple[int, int], value: int, tries: int = 3) -> None:
        """Write and confirm. EEPROM writes can reply late or not at all, so a
        missing reply is settled by reading the register back."""
        addr, size = reg
        for i in range(tries):
            self._send(sid, WRITE, bytes([addr]) + int(value).to_bytes(size, "little"))
            try:
                self._status()
                return
            except BusError:
                time.sleep(0.02)
                try:
                    if self.read(sid, reg) == int(value):
                        return
                except BusError:
                    pass
        raise BusError(f"write of {value} to {addr} on servo {sid} not confirmed")

    # -- many servos, one packet ----------------------------------------------

    def sync_write(self, reg: tuple[int, int], values: dict[int, int]) -> None:
        addr, size = reg
        params = bytes([addr, size])
        for sid, v in values.items():
            params += bytes([sid]) + int(v).to_bytes(size, "little")
        self._send(0xFE, SYNC_WRITE, params)  # broadcast: no reply

    def sync_read(self, reg: tuple[int, int], ids) -> dict[int, int]:
        addr, size = reg
        ids = list(ids)
        self._send(0xFE, SYNC_READ, bytes([addr, size]) + bytes(ids))
        out = {}
        for _ in ids:
            sid, _, p = self._status()
            out[sid] = int.from_bytes(p[:size], "little")
        if set(out) != set(ids):
            raise BusError(f"sync read missed {set(ids) - set(out)}")
        return out

    def read_retry(self, sid: int, reg, tries: int = 3) -> int:
        for i in range(tries):
            try:
                return self.read(sid, reg)
            except BusError:
                if i == tries - 1:
                    raise
                time.sleep(0.005)
        raise AssertionError
