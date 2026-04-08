import argparse
import binascii
import os
import sys

try:
    from scapy.all import Ether, PcapReader  # type: ignore
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: 'scapy'. Install it with `python -m pip install scapy` "
        "(or from `requirements.txt`)."
    ) from e


PTP_ETHERTYPE = 0x88F7  # 35063

# Offsets assumed by Scripts/AnnouncePTP.py and Scripts/Announce_Attack.py
REQUIRED_RANGES = {
    "ptp_header_first_byte": (0, 1),
    "clockIdentity": (20, 28),
    "sequenceId": (30, 32),
    "priority1": (47, 48),
    "grandmasterClockClass": (48, 49),
    "priority2": (52, 53),
    "grandmasterClockIdentity": (53, 61),
    "localStepsRemoved": (61, 63),
    "timeSource": (63, 64),
}


def slice_ok(payload: bytes, start: int, end: int) -> bool:
    return len(payload) >= end and start >= 0 and end >= start


def hx(b: bytes) -> str:
    return binascii.hexlify(b).decode("ascii")


def format_mac(mac: str | None) -> str:
    return mac or "<unknown>"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that Announce packets in a PCAP match the byte-offset assumptions "
            "used by the Announce attack scripts."
        )
    )
    parser.add_argument(
        "pcap",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "ptp_server.pcap"),
        help="Path to the .pcap to validate (default: ptp_server.pcap next to this script).",
    )
    parser.add_argument(
        "--print-announce",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Print details of the Nth Announce frame (0-based among Announce frames only), "
            "to compare against Wireshark."
        ),
    )
    parser.add_argument(
        "--print-packet-index",
        type=int,
        default=None,
        metavar="IDX",
        help="Print details of the packet at absolute PCAP packet index IDX (0-based).",
    )
    parser.add_argument(
        "--hexdump-bytes",
        type=int,
        default=96,
        metavar="BYTES",
        help="How many bytes of the PTP payload to hex-dump (default: 96).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pcap):
        print(f"PCAP not found: {args.pcap}", file=sys.stderr)
        return 2

    total_ptp = 0
    total_announce = 0
    announce_transport_specific = set()
    bad_packets: list[tuple[int, int, str]] = []
    announce_counter = -1
    printed = False

    with PcapReader(args.pcap) as pcap:
        for idx, pkt in enumerate(pcap):
            if Ether not in pkt:
                continue
            if pkt[Ether].type != PTP_ETHERTYPE:
                continue
            if not hasattr(pkt, "load"):
                continue

            payload: bytes = bytes(pkt.load)
            if not payload:
                continue

            total_ptp += 1

            # IEEE1588: first byte = (transportSpecific << 4) | messageType
            first = payload[0]
            transport_specific = (first >> 4) & 0x0F
            message_type = first & 0x0F

            want_print = False
            if args.print_packet_index is not None and idx == args.print_packet_index:
                want_print = True
            if message_type == 11:
                announce_counter += 1
                if args.print_announce is not None and announce_counter == args.print_announce:
                    want_print = True

            if want_print and not printed:
                eth = pkt[Ether]
                print("\n=== Frame details ===")
                print(f"pcap_packet_index: {idx}")
                if message_type == 11:
                    print(f"announce_index: {announce_counter}")
                print(f"Ether.src: {format_mac(getattr(eth, 'src', None))}")
                print(f"Ether.dst: {format_mac(getattr(eth, 'dst', None))}")
                print(f"Ether.type: 0x{int(eth.type):04x}")
                print(f"ptp.payload_len: {len(payload)}")
                print(f"ptp.first_byte: 0x{first:02x}")
                print(f"ptp.transportSpecific: {transport_specific}")
                print(f"ptp.messageType(lowNibble): {message_type}")

                # Print key fields used by the attack (only if present)
                def show_field(name: str) -> None:
                    start, end = REQUIRED_RANGES[name]
                    if slice_ok(payload, start, end):
                        val = payload[start:end]
                        extra = ""
                        if name == "sequenceId":
                            extra = f" (uint16={int.from_bytes(val, 'big')})"
                        print(f"{name}[{start}:{end}]: {hx(val)}{extra}")
                    else:
                        print(f"{name}[{start}:{end}]: <missing>")

                for field in [
                    "clockIdentity",
                    "sequenceId",
                    "priority1",
                    "grandmasterClockClass",
                    "priority2",
                    "grandmasterClockIdentity",
                    "localStepsRemoved",
                    "timeSource",
                ]:
                    show_field(field)

                dump_n = max(0, int(args.hexdump_bytes))
                dumped = payload[:dump_n]
                print(f"\nptp.payload_hexdump_first_{dump_n}_bytes:\n{hx(dumped)}")
                print("=== End frame details ===\n")
                printed = True

            if message_type != 11:  # Announce
                continue

            total_announce += 1
            announce_transport_specific.add(transport_specific)

            # Validate all required ranges exist
            for name, (start, end) in REQUIRED_RANGES.items():
                if not slice_ok(payload, start, end):
                    bad_packets.append((idx, len(payload), name))

    print(f"PCAP: {args.pcap}")
    print(f"PTP EtherType frames: {total_ptp}")
    print(f"Announce frames (messageType=11): {total_announce}")
    if total_announce:
        ts_sorted = ", ".join(str(x) for x in sorted(announce_transport_specific))
        print(f"Announce transportSpecific seen: {ts_sorted}")

    if total_announce == 0:
        print(
            "No Announce frames found. If you expect Announce traffic, verify the capture "
            "contains PTP over Ethernet (0x88F7) and that messageType parsing is correct.",
            file=sys.stderr,
        )
        return 3

    if bad_packets:
        print("\nFAIL: Some Announce frames are too short for required offsets.", file=sys.stderr)
        # Show a compact summary (first 25)
        for idx, length, missing in bad_packets[:25]:
            print(f"- packet_index={idx} payload_len={length} missing={missing}", file=sys.stderr)
        if len(bad_packets) > 25:
            print(f"... and {len(bad_packets) - 25} more", file=sys.stderr)
        return 1

    print("\nOK: All Announce frames contain the required byte offsets for the attack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

