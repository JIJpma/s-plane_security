import os
import sys

try:
    from scapy.all import Ether, PcapReader
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: 'scapy'. "
        "Install it with `python -m pip install scapy` (or from `requirements.txt`)."
    ) from e

# When running `python main.py` from this folder, Python's import path won't
# include the parent `PipelineTestAttacker/` directory where `Scripts/` lives.
# Add it so `from Scripts.* import ...` works.
HERE = os.path.dirname(__file__)
PIPELINE_TEST_ATTACKER_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PIPELINE_TEST_ATTACKER_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_TEST_ATTACKER_ROOT)

if __name__ == "__main__":
    # Local imports to keep this script importable even when not used.
    from Scripts.AnnouncePTP import AnnouncePTP  # noqa: F401
    from Scripts.SyncPTP import SyncPTP  # noqa: F401

    # Load ptp pcap data
    default_pcap = os.path.join(HERE, "ptp_server.pcap")
    file = (
        default_pcap
        if os.path.exists(default_pcap)
        else "Scripts/test_ptp_packets/ptp_packets.pcap"
    )
    pcap_reader = PcapReader(file)
    for packet in pcap_reader:
        if Ether in packet:
            if packet[Ether].type == 35063:
                message_type = packet[0].load[0]
                if message_type == 11:
                    print("Announce packet")
                elif message_type == 0:
                    print("Sync packet")
                elif message_type == 8:
                    print("Follow_Up packet")
                elif message_type == 1:
                    print("Delay_Req packet")
                elif message_type == 9:
                    print("Delay_Resp packet")
                else:
                    print(f"Unknown packet: {message_type}")